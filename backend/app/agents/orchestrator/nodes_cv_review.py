"""
CV-only review flow nodes for the orchestrator graph.

Decomposes the monolithic ``RESUME_REFINEMENT_SYSTEM_PROMPT`` (previously
prepended to every Writer turn for the cv_only flow) into six deterministic
graph nodes, each followed by a structured approval gate:

    1. Header               -> approve / edit / reject
    2. Education            -> approve / edit / reject
    3. Experience           -> approve / edit / reject
    4. Leadership & activities (skipped automatically when empty)
                            -> approve / edit / reject
    5. Skills & projects    -> approve / edit / reject
    6. Overall assessment   -> approve / reject  (no further work after this)

Each step node calls ``ChatOpenAI(...).with_structured_output(SchemaCls)`` once
with a tightly scoped per-step prompt extracted from the original system
prompt. The LLM only ever sees the data for that section, so reviews stay
focused and tokens stay bounded.

Per-step outputs land in ``state['review_outputs'][step]`` so the API layer
can return the whole review at the end without re-running anything.

Edit gates fold the user's feedback into the next call by appending it to the
"feedback so far" block of the per-step prompt before re-invoking the LLM.

Notes on testing:
The module-level imports of ``chat_model`` and the schemas are intentional
so unit tests can ``monkeypatch.setattr(nodes_cv_review.chat_model, ...)``
to inject deterministic stubs without touching ``langchain_openai`` directly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ConfigDict

from app.agents.orchestrator.gates import GatePayload, GateResolution
from app.agents.orchestrator.outputs import RenderedCv, file_record, session_output_dir, with_file
from app.agents.orchestrator.state import OrchestratorState
from app.agents.writer_agent import generate_cv_pdf
from app.openai_llm import chat_model


# --------------------------------------------------------------------------- #
# Node-name constants
# --------------------------------------------------------------------------- #

CR_ENTRY = "cr_entry"
CR_STEP1_HEADER = "cr_step1_header"
CR_GATE_HEADER = "cr_gate_header"
CR_STEP2_EDUCATION = "cr_step2_education"
CR_GATE_EDUCATION = "cr_gate_education"
CR_STEP3_EXPERIENCE = "cr_step3_experience"
CR_GATE_EXPERIENCE = "cr_gate_experience"
CR_STEP4_LEADERSHIP = "cr_step4_leadership"
CR_GATE_LEADERSHIP = "cr_gate_leadership"
CR_STEP5_SKILLS_PROJECTS = "cr_step5_skills_projects"
CR_GATE_SKILLS_PROJECTS = "cr_gate_skills_projects"
CR_STEP6_ASSESSMENT = "cr_step6_assessment"
CR_GATE_ASSESSMENT = "cr_gate_assessment"
CR_STEP7_FINALIZE = "cr_step7_finalize_cv"

# Kept for backwards compatibility with the earlier stub. The graph builder
# uses CR_ENTRY now; CR_STUB is intentionally not registered as a node any
# more but the constant is preserved so external imports do not break.
CR_STUB = CR_ENTRY


# --------------------------------------------------------------------------- #
# Shared output shape primitives
# --------------------------------------------------------------------------- #

Verdict = Literal["Strong", "Acceptable", "Weak"]


class HeaderReview(BaseModel):
    verdict: Verdict = Field(description="Overall verdict on the header.")
    critique: list[str] = Field(
        default_factory=list,
        description="Concrete observations about what is missing, weak, or inconsistent.",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable improvement suggestions, e.g. 'Add GitHub URL'.",
    )
    rewritten_header: str | None = Field(
        default=None,
        description=(
            "A rewritten one-block header (name, location, email, links) if "
            "improvements are warranted. Null if the original is fine."
        ),
    )


class EducationReview(BaseModel):
    critique: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    improved_entries: list[str] = Field(
        default_factory=list,
        description=(
            "Rewritten education entries in the same order as the input. "
            "Empty list when no rewrite is needed."
        ),
    )
    future_degree_recommendation: str | None = Field(
        default=None,
        description="One-sentence note on whether/which future degrees make sense.",
    )


class ExperienceBulletRewrite(BaseModel):
    role_index: int = Field(
        description=(
            "0-based index into the original experience array (or -1 if the "
            "bullet is a free-floating string)."
        )
    )
    original: str
    rewritten: str
    rationale: str | None = Field(
        default=None,
        description="Why the rewrite is stronger (verb, metric, scope, etc.).",
    )


class ExperienceReview(BaseModel):
    role_critiques: list[str] = Field(
        default_factory=list,
        description="One short critique per role (relevance, action verbs, metrics).",
    )
    rewritten_bullets: list[ExperienceBulletRewrite] = Field(
        default_factory=list,
        description="Improved bullets. Only include bullets you actually rewrote.",
    )
    weak_bullets_to_remove: list[str] = Field(
        default_factory=list,
        description="Bullets that should be cut entirely.",
    )


class LeadershipReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    critique: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    activities_to_emphasize: list[str] = Field(default_factory=list)
    activities_to_remove: list[str] = Field(default_factory=list)


class SkillGroup(BaseModel):
    """One named skill category. Replaces ``dict[str, list[str]]`` single-key dicts
    so OpenAI structured output gets ``additionalProperties: false``."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Category label, e.g. Languages, Frameworks, ML")
    skills: list[str] = Field(
        default_factory=list,
        description="Skills in this category, ordered by relevance.",
    )


class RefinedProjectDescription(BaseModel):
    """Rewritten description for one project."""
    model_config = ConfigDict(extra="forbid")

    project: str = Field(description="Project title or short label")
    description: str = Field(description="Refined description emphasising architecture and impact")


class SkillsProjectsReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_groups: list[SkillGroup] = Field(
        default_factory=list,
        description="Re-grouped skills, ordered by relevance.",
    )
    skills_to_remove: list[str] = Field(default_factory=list)
    project_ranking: list[str] = Field(
        default_factory=list,
        description="Project titles or short labels ordered from strongest to weakest.",
    )
    refined_project_descriptions: list[RefinedProjectDescription] = Field(
        default_factory=list,
        description="Rewritten descriptions for the top projects (up to 4).",
    )
    projects_to_remove: list[str] = Field(default_factory=list)


class OverallAssessment(BaseModel):
    score: float = Field(ge=0, le=10, description="Resume score 0-10.")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    top_improvements: list[str] = Field(
        default_factory=list,
        description="The three highest-impact next steps for the candidate.",
    )
    recommended_pages: Literal[1, 2] = Field(
        default=1,
        description="Whether the resume should ultimately be 1 or 2 pages.",
    )
    role_specific_tailoring: str | None = Field(
        default=None,
        description="Optional note on role-specific tailoring (AI vs. SWE vs. research).",
    )


# --------------------------------------------------------------------------- #
# Shared prompt preamble
# Trimmed-down version of RESUME_REFINEMENT_SYSTEM_PROMPT's GLOBAL OBJECTIVES
# section. Kept short on purpose - per-step prompts add the rules for that step.
# --------------------------------------------------------------------------- #

_GLOBAL_PREAMBLE = (
    "You are a senior technical recruiter and resume optimisation expert "
    "specialising in software engineering, AI, and data roles. "
    "Optimise for recruiter skim-reading and ATS compatibility. "
    "Prefer impact-driven wording with concrete metrics over task descriptions. "
    "Preserve a professional, neutral tone - do not inflate seniority or fabricate "
    "experience, metrics, or technologies. Use a European/German academic and "
    "industry frame of reference unless the resume says otherwise."
)


def _format_user_block(section_label: str, payload: Any, feedback: list[str]) -> str:
    """
    Build the user-turn content for a step.

    Includes the JSON-encoded original section data and any accumulated user
    feedback from prior edit gates for this step.
    """
    blocks: list[str] = [
        f"Original {section_label} (verbatim from the resume):",
        "```json",
        json.dumps(payload, indent=2, default=str),
        "```",
    ]
    if feedback:
        blocks.append("")
        blocks.append("User feedback from previous review passes:")
        for i, fb in enumerate(feedback, 1):
            blocks.append(f"{i}. {fb}")
    return "\n".join(blocks)


def _run_step_review(
    schema_cls: type[BaseModel],
    system_text: str,
    section_label: str,
    payload: Any,
    feedback: list[str],
) -> dict[str, Any]:
    """
    Invoke ChatOpenAI with structured output and return the model as a dict.

    Tests monkeypatch this whole function to avoid hitting the network. Keeping
    it as a module-level callable rather than inlining the chain construction
    makes that monkeypatch a single seam.
    """
    llm = chat_model("writer", temperature=0.2)
    structured = llm.with_structured_output(schema_cls)
    user_content = _format_user_block(section_label, payload, feedback)
    result = structured.invoke(
        [
            {"role": "system", "content": _GLOBAL_PREAMBLE + "\n\n" + system_text},
            {"role": "user", "content": user_content},
        ]
    )
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {}


def _narrate(state_update: dict[str, Any], text: str) -> dict[str, Any]:
    """Same pattern as nodes_job_tailoring._narrate - kept local to avoid coupling."""
    state_update["last_narration"] = text
    state_update.setdefault("messages", []).append(AIMessage(content=text))
    state_update["pending_gate"] = None
    return state_update


def _record_review(state: OrchestratorState, step: str, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge a per-step review output into ``state['review_outputs']`` immutably."""
    existing = dict(state.get("review_outputs") or {})
    existing[step] = payload
    return existing


def _feedback_for(state: OrchestratorState, step: str) -> list[str]:
    """Pull the accumulated edit-loop feedback for a given step out of state."""
    feedback_bucket = (state.get("review_outputs") or {}).get("_feedback") or {}
    return list(feedback_bucket.get(step, []))


def _append_feedback(state: OrchestratorState, step: str, text: str) -> dict[str, dict[str, Any]]:
    """Persist new edit-gate feedback so the next pass of the step picks it up."""
    existing = dict(state.get("review_outputs") or {})
    bucket = dict(existing.get("_feedback") or {})
    bucket[step] = list(bucket.get(step, [])) + [text]
    existing["_feedback"] = bucket
    return existing


# --------------------------------------------------------------------------- #
# Per-step prompts
# Distilled from the corresponding section of RESUME_REFINEMENT_SYSTEM_PROMPT.
# --------------------------------------------------------------------------- #

_HEADER_PROMPT = """
STEP 1: HEADER REVIEW

Evaluate the header for:
- Essential information (name, email, location).
- Missing high-impact links (GitHub, LinkedIn, portfolio).
- Professional formatting and conciseness.

Return:
- verdict: Strong / Acceptable / Weak.
- critique: short, specific observations.
- suggestions: concrete improvements.
- rewritten_header: a single rewritten header block if improvements are warranted.

Do NOT fabricate links or contact details that are not in the original.
""".strip()


_EDUCATION_PROMPT = """
STEP 2: EDUCATION SECTION REVIEW

Evaluate:
- Clarity of degree status (completed, ongoing, planned).
- Whether grades are used appropriately or feel unnecessary.
- Whether academic strengths are underutilised.

Refine by:
- Clarifying timelines.
- Adding relevant coursework only when it strengthens the profile.
- Removing speculative or confusing entries.

Return improved_entries in the same order as the input list. Empty list = no rewrite needed.
Add a short future_degree_recommendation if relevant.
""".strip()


_EXPERIENCE_PROMPT = """
STEP 3: EXPERIENCE SECTION REVIEW

For each role:
- Evaluate relevance to software / AI / data roles.
- Identify bullets that are descriptive instead of impact-driven.
- Flag bullets that lack measurable outcomes (numbers, percentages, scale, scope).
- Check the action-technology-outcome structure.

Rewrite bullets to:
- Start with strong action verbs.
- Mention technologies naturally.
- Highlight contribution, responsibility, or outcome with a concrete metric where possible.

Return:
- role_critiques: at most one short bullet per role.
- rewritten_bullets: only the bullets you actually rewrote, with original + rewritten + rationale.
- weak_bullets_to_remove: any bullets that should be cut entirely.

Do NOT invent metrics, technologies, scope, or outcomes that are not implied by the original.
""".strip()


_LEADERSHIP_PROMPT = """
STEP 4: LEADERSHIP & ACTIVITIES REVIEW

Evaluate:
- Relevance and impact of leadership roles.
- Clarity of extracurricular activities and volunteer work.
- Whether activities demonstrate valuable, transferable skills.
- Overlap or redundancy between activities.

Refine by:
- Emphasising leadership impact and responsibilities.
- Highlighting transferable skills.
- Clarifying scope and significance.
- Removing weak or irrelevant entries.

Return activities_to_emphasize and activities_to_remove using verbatim activity titles or
short labels that uniquely identify the activity in the input list.
""".strip()


_SKILLS_PROJECTS_PROMPT = """
STEP 5: SKILLS & PROJECTS REVIEW

SKILLS:
- Evaluate skill relevance, readability, and grouping.
- Identify "shopping list" anti-patterns.
- Re-group into logical categories ordered by relevance.
- Remove redundancy or weak signals.

PROJECTS:
- Evaluate technical depth, ownership, complexity, and system thinking.
- Identify overlap or redundancy between projects.
- Rank projects strongest -> weakest.
- For the top projects (up to 4), provide refined descriptions emphasising
  architecture and intelligence rather than feature lists.
- Recommend projects_to_remove if the section is too long.

Return skill_groups as a list of objects: [{"name": "Languages", "skills": ["Python", "Go"]}, ...].
""".strip()


_ASSESSMENT_PROMPT = """
STEP 6: OVERALL ASSESSMENT

Synthesise the prior section reviews into a final assessment.

Provide:
- score: 0-10.
- strengths: short bullets.
- weaknesses: short bullets.
- top_improvements: the three highest-impact next steps.
- recommended_pages: 1 or 2.
- role_specific_tailoring: a short note if the candidate would benefit from
  tailoring the resume to a specific role family (AI, SWE, research, data).
""".strip()


# --------------------------------------------------------------------------- #
# Entry node
# --------------------------------------------------------------------------- #

def cr_entry(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    name = cv_data.get("name", "you")
    sections = [
        "Header",
        "Education",
        "Experience",
    ]
    if cv_data.get("leadership_activities"):
        sections.append("Leadership & activities")
    sections.append("Skills & projects")
    sections.append("Overall assessment")
    narration = (
        f"Starting the CV review for {name}. I'll work section by section "
        f"({len(sections)} steps): " + " -> ".join(sections) + ". "
        "After each section I'll pause for your approval; if you want changes, "
        "tell me what to adjust and I'll re-run that section."
    )
    return _narrate({}, narration)


# --------------------------------------------------------------------------- #
# Step 1: Header
# --------------------------------------------------------------------------- #

def _header_payload(cv_data: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "email", "phone", "location", "github_url", "linkedin_url", "portfolio_url")
    return {k: cv_data.get(k) for k in keys}


def cr_step1_header(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    payload = _header_payload(cv_data)
    review = _run_step_review(
        HeaderReview,
        _HEADER_PROMPT,
        "header data",
        payload,
        _feedback_for(state, "header"),
    )

    verdict = review.get("verdict", "Acceptable")
    suggestions = review.get("suggestions") or []
    narration_lines = [
        "Header review",
        "",
        f"Verdict: {verdict}",
    ]
    if suggestions:
        narration_lines.append("Suggestions:")
        narration_lines.extend(f"- {s}" for s in suggestions[:5])
    if review.get("rewritten_header"):
        narration_lines.append("")
        narration_lines.append("Proposed rewrite:")
        narration_lines.append(review["rewritten_header"])

    return _narrate(
        {"review_outputs": _record_review(state, "header", review)},
        "\n".join(narration_lines),
    )


def cr_gate_header(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("header") or {}
    payload = GatePayload(
        step="cv_review_header",
        kind="approval",
        narration="Approve the header review, edit with feedback, or reject to stop.",
        preview=review,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        return Command(update={"pending_gate": None}, goto=CR_STEP2_EDUCATION)
    if resolution.action == "edit":
        return Command(
            update=_narrate(
                {"review_outputs": _append_feedback(state, "header", resolution.feedback or "")},
                f"Re-running the header review with your feedback: {resolution.feedback}",
            ),
            goto=CR_STEP1_HEADER,
        )
    return Command(update=_narrate({}, "Stopped during header review."), goto=END)


# --------------------------------------------------------------------------- #
# Step 2: Education
# --------------------------------------------------------------------------- #

def cr_step2_education(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    payload = cv_data.get("education") or []
    review = _run_step_review(
        EducationReview,
        _EDUCATION_PROMPT,
        "education entries",
        payload,
        _feedback_for(state, "education"),
    )

    suggestions = review.get("suggestions") or []
    improved = review.get("improved_entries") or []
    lines = [
        "Education review",
        "",
        f"Suggestions: {len(suggestions)} | Improved entries: {len(improved)}",
    ]
    if review.get("future_degree_recommendation"):
        lines.append("")
        lines.append(f"On future degrees: {review['future_degree_recommendation']}")

    return _narrate(
        {"review_outputs": _record_review(state, "education", review)},
        "\n".join(lines),
    )


def cr_gate_education(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("education") or {}
    payload = GatePayload(
        step="cv_review_education",
        kind="approval",
        narration="Approve the education review, edit with feedback, or reject to stop.",
        preview=review,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        return Command(update={"pending_gate": None}, goto=CR_STEP3_EXPERIENCE)
    if resolution.action == "edit":
        return Command(
            update=_narrate(
                {"review_outputs": _append_feedback(state, "education", resolution.feedback or "")},
                f"Re-running the education review with your feedback: {resolution.feedback}",
            ),
            goto=CR_STEP2_EDUCATION,
        )
    return Command(update=_narrate({}, "Stopped during education review."), goto=END)


# --------------------------------------------------------------------------- #
# Step 3: Experience
# --------------------------------------------------------------------------- #

def cr_step3_experience(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    payload = cv_data.get("experience") or []
    review = _run_step_review(
        ExperienceReview,
        _EXPERIENCE_PROMPT,
        "experience section",
        payload,
        _feedback_for(state, "experience"),
    )

    rewritten = review.get("rewritten_bullets") or []
    removed = review.get("weak_bullets_to_remove") or []
    lines = [
        "Experience review",
        "",
        f"Bullets rewritten: {len(rewritten)} | Bullets flagged for removal: {len(removed)}",
    ]
    return _narrate(
        {"review_outputs": _record_review(state, "experience", review)},
        "\n".join(lines),
    )


def cr_gate_experience(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("experience") or {}
    payload = GatePayload(
        step="cv_review_experience",
        kind="approval",
        narration="Approve the experience review, edit with feedback, or reject to stop.",
        preview=review,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        # Skip leadership step if the CV has no leadership_activities section.
        cv_data = state.get("cv_data") or {}
        if cv_data.get("leadership_activities"):
            return Command(update={"pending_gate": None}, goto=CR_STEP4_LEADERSHIP)
        return Command(update={"pending_gate": None}, goto=CR_STEP5_SKILLS_PROJECTS)
    if resolution.action == "edit":
        return Command(
            update=_narrate(
                {"review_outputs": _append_feedback(state, "experience", resolution.feedback or "")},
                f"Re-running the experience review with your feedback: {resolution.feedback}",
            ),
            goto=CR_STEP3_EXPERIENCE,
        )
    return Command(update=_narrate({}, "Stopped during experience review."), goto=END)


# --------------------------------------------------------------------------- #
# Step 4: Leadership & activities (conditionally present)
# --------------------------------------------------------------------------- #

def cr_step4_leadership(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    payload = cv_data.get("leadership_activities") or []
    review = _run_step_review(
        LeadershipReview,
        _LEADERSHIP_PROMPT,
        "leadership & activities section",
        payload,
        _feedback_for(state, "leadership"),
    )

    emphasise = review.get("activities_to_emphasize") or []
    remove = review.get("activities_to_remove") or []
    lines = [
        "Leadership & activities review",
        "",
        f"Emphasise: {len(emphasise)} | Remove: {len(remove)}",
    ]
    return _narrate(
        {"review_outputs": _record_review(state, "leadership", review)},
        "\n".join(lines),
    )


def cr_gate_leadership(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("leadership") or {}
    payload = GatePayload(
        step="cv_review_leadership",
        kind="approval",
        narration="Approve the leadership review, edit with feedback, or reject to stop.",
        preview=review,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        return Command(update={"pending_gate": None}, goto=CR_STEP5_SKILLS_PROJECTS)
    if resolution.action == "edit":
        return Command(
            update=_narrate(
                {"review_outputs": _append_feedback(state, "leadership", resolution.feedback or "")},
                f"Re-running the leadership review with your feedback: {resolution.feedback}",
            ),
            goto=CR_STEP4_LEADERSHIP,
        )
    return Command(update=_narrate({}, "Stopped during leadership review."), goto=END)


# --------------------------------------------------------------------------- #
# Step 5: Skills & projects
# --------------------------------------------------------------------------- #

def cr_step5_skills_projects(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    payload = {
        "skills": cv_data.get("skills") or [],
        "projects": cv_data.get("projects") or [],
    }
    review = _run_step_review(
        SkillsProjectsReview,
        _SKILLS_PROJECTS_PROMPT,
        "skills & projects sections",
        payload,
        _feedback_for(state, "skills_projects"),
    )

    project_ranking = review.get("project_ranking") or []
    refined = review.get("refined_project_descriptions") or []
    lines = [
        "Skills & projects review",
        "",
        f"Skill groups: {len(review.get('skill_groups') or [])} | "
        f"Skills to remove: {len(review.get('skills_to_remove') or [])}",
        f"Projects ranked: {len(project_ranking)} | Refined descriptions: {len(refined)} | "
        f"Projects to remove: {len(review.get('projects_to_remove') or [])}",
    ]
    return _narrate(
        {"review_outputs": _record_review(state, "skills_projects", review)},
        "\n".join(lines),
    )


def cr_gate_skills_projects(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("skills_projects") or {}
    payload = GatePayload(
        step="cv_review_skills_projects",
        kind="approval",
        narration="Approve the skills & projects review, edit with feedback, or reject to stop.",
        preview=review,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        return Command(update={"pending_gate": None}, goto=CR_STEP6_ASSESSMENT)
    if resolution.action == "edit":
        return Command(
            update=_narrate(
                {"review_outputs": _append_feedback(state, "skills_projects", resolution.feedback or "")},
                f"Re-running the skills & projects review with your feedback: {resolution.feedback}",
            ),
            goto=CR_STEP5_SKILLS_PROJECTS,
        )
    return Command(update=_narrate({}, "Stopped during skills & projects review."), goto=END)


# --------------------------------------------------------------------------- #
# Step 6: Overall assessment
# --------------------------------------------------------------------------- #

def cr_step6_assessment(state: OrchestratorState) -> dict[str, Any]:
    # The assessment is synthesised from the prior section reviews so the LLM
    # has the full critique map as context.
    payload = {
        "section_reviews": {
            k: v for k, v in (state.get("review_outputs") or {}).items() if not k.startswith("_")
        },
        "cv_summary": {
            "name": (state.get("cv_data") or {}).get("name"),
            "education_count": len((state.get("cv_data") or {}).get("education") or []),
            "experience_count": len((state.get("cv_data") or {}).get("experience") or []),
            "skills_count": len((state.get("cv_data") or {}).get("skills") or []),
            "projects_count": len((state.get("cv_data") or {}).get("projects") or []),
        },
    }
    review = _run_step_review(
        OverallAssessment,
        _ASSESSMENT_PROMPT,
        "complete prior reviews",
        payload,
        _feedback_for(state, "assessment"),
    )

    score = review.get("score")
    top = review.get("top_improvements") or []
    lines = [
        "Overall assessment",
        "",
        f"Score: {score:.1f} / 10" if isinstance(score, (int, float)) else "Score: n/a",
        f"Recommended length: {review.get('recommended_pages', 1)} page(s)",
    ]
    if top:
        lines.append("")
        lines.append("Top improvements:")
        lines.extend(f"- {t}" for t in top[:5])
    if review.get("role_specific_tailoring"):
        lines.append("")
        lines.append(f"Role-specific tailoring: {review['role_specific_tailoring']}")

    return _narrate(
        {"review_outputs": _record_review(state, "assessment", review)},
        "\n".join(lines),
    )


def cr_gate_assessment(state: OrchestratorState) -> Command:
    review = (state.get("review_outputs") or {}).get("assessment") or {}
    payload = GatePayload(
        step="cv_review_assessment",
        kind="approval",
        narration="Approve to finish the review. Reject to discard the assessment.",
        preview=review,
        allowed_actions=["approve", "reject"],
    ).model_dump()
    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    if resolution.action == "approve":
        return Command(update={"pending_gate": None}, goto=CR_STEP7_FINALIZE)
    return Command(update=_narrate({}, "CV review ended without a final assessment."), goto=END)


# --------------------------------------------------------------------------- #
# Step 7: assemble the revised CV from approved section reviews and export a
# downloadable PDF. Runs automatically once the assessment gate is approved;
# no further gate follows.
# --------------------------------------------------------------------------- #

def _extract_filename(tool_message: str) -> str | None:
    """Pull a filename out of a generate_* tool success message."""
    match = re.search(r"The file '([^']+)' is ready for download", tool_message or "")
    return match.group(1) if match else None


def _apply_education_review(education: list[Any], review: dict[str, Any]) -> list[Any]:
    improved = [e for e in (review.get("improved_entries") or []) if isinstance(e, str) and e.strip()]
    # Rewritten entries are flat strings; only swap them in when the original
    # entries are also strings, otherwise we'd destroy the structured
    # institution/degree/dates fields the LaTeX template relies on.
    if (
        improved
        and len(improved) == len(education)
        and all(isinstance(e, str) for e in education)
    ):
        return improved
    return list(education)


def _apply_experience_review(experience: list[Any], review: dict[str, Any]) -> list[Any]:
    rewrites: dict[str, str] = {}
    for item in review.get("rewritten_bullets") or []:
        if not isinstance(item, dict):
            continue
        original, rewritten = item.get("original"), item.get("rewritten")
        if isinstance(original, str) and isinstance(rewritten, str) and original.strip():
            rewrites[original.strip()] = rewritten
    to_remove = {
        text.strip() for text in (review.get("weak_bullets_to_remove") or []) if isinstance(text, str)
    }
    if not rewrites and not to_remove:
        return list(experience)

    new_experience = []
    for role in experience:
        if not isinstance(role, dict):
            new_experience.append(role)
            continue
        role = dict(role)
        new_responsibilities = []
        for bullet in role.get("responsibilities") or []:
            if not isinstance(bullet, str):
                new_responsibilities.append(bullet)
                continue
            key = bullet.strip()
            if key in to_remove:
                continue
            new_responsibilities.append(rewrites.get(key, bullet))
        role["responsibilities"] = new_responsibilities
        new_experience.append(role)
    return new_experience


def _activity_label(activity: Any) -> str:
    """Comparable text for an activity: the string itself, or role+organization
    for structured LeadershipEntry dicts."""
    if isinstance(activity, str):
        return activity.strip().lower()
    if isinstance(activity, dict):
        parts = [activity.get("role"), activity.get("organization"), activity.get("description")]
        return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip()).lower()
    return ""


def _apply_leadership_review(activities: list[Any], review: dict[str, Any]) -> list[Any]:
    emphasize = [a for a in (review.get("activities_to_emphasize") or []) if isinstance(a, str) and a.strip()]
    remove = [a.strip().lower() for a in (review.get("activities_to_remove") or []) if isinstance(a, str) and a.strip()]

    def _is_removed(activity: Any) -> bool:
        label = _activity_label(activity)
        return bool(label) and any(r in label or label in r for r in remove)

    kept = [a for a in activities if not _is_removed(a)]
    if not emphasize:
        return kept

    ordered: list[Any] = []
    remaining = list(kept)
    for label in emphasize:
        lbl = label.strip().lower()
        for activity in list(remaining):
            activity_label = _activity_label(activity)
            if activity_label and (lbl in activity_label or activity_label in lbl):
                ordered.append(activity)
                remaining.remove(activity)
                break
    ordered.extend(remaining)
    return ordered


def _apply_skills_projects_review(
    skills: list[Any],
    projects: list[Any],
    review: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    skill_groups = review.get("skill_groups") or []
    to_remove_skills = {
        s.strip().lower() for s in (review.get("skills_to_remove") or []) if isinstance(s, str)
    }

    # Preserve the reviewer's grouping: emit JSON Resume-style skill dicts
    # ({"name", "keywords"}) that the LaTeX adapter renders as labelled rows.
    new_skills: list[Any] = []
    if skill_groups:
        for group in skill_groups:
            if not isinstance(group, dict):
                continue
            kept = [
                skill for skill in (group.get("skills") or [])
                if isinstance(skill, str) and skill.strip().lower() not in to_remove_skills
            ]
            if kept:
                new_skills.append({"name": group.get("name") or "Skills", "keywords": kept})
    if not new_skills:
        new_skills = [s for s in skills if not (isinstance(s, str) and s.strip().lower() in to_remove_skills)]

    projects_to_remove = {
        p.strip().lower() for p in (review.get("projects_to_remove") or []) if isinstance(p, str)
    }
    refined = {
        item.get("project", "").strip().lower(): item.get("description")
        for item in (review.get("refined_project_descriptions") or [])
        if isinstance(item, dict) and isinstance(item.get("project"), str)
    }
    ranking = [p.strip().lower() for p in (review.get("project_ranking") or []) if isinstance(p, str)]

    def _project_name(proj: Any) -> str:
        return str(proj.get("name") or "").strip().lower() if isinstance(proj, dict) else ""

    filtered_projects = []
    for proj in projects:
        name_key = _project_name(proj)
        if name_key and any(r in name_key or name_key in r for r in projects_to_remove):
            continue
        if isinstance(proj, dict):
            proj = dict(proj)
            for r_name, r_desc in refined.items():
                if r_name and (r_name in name_key or name_key in r_name) and isinstance(r_desc, str) and r_desc.strip():
                    proj["description"] = r_desc
                    break
        filtered_projects.append(proj)

    if ranking:
        def _rank_key(proj: Any) -> int:
            name_key = _project_name(proj)
            for idx, r in enumerate(ranking):
                if r and (r in name_key or name_key in r):
                    return idx
            return len(ranking)
        filtered_projects.sort(key=_rank_key)

    return new_skills, filtered_projects


def _assemble_reviewed_cv(state: OrchestratorState) -> dict[str, Any]:
    """Fold the approved per-section reviews back into the original cv_data."""
    cv_data = dict(state.get("cv_data") or {})
    reviews = state.get("review_outputs") or {}

    if cv_data.get("education"):
        cv_data["education"] = _apply_education_review(cv_data["education"], reviews.get("education") or {})
    if cv_data.get("experience"):
        cv_data["experience"] = _apply_experience_review(cv_data["experience"], reviews.get("experience") or {})
    if cv_data.get("leadership_activities"):
        cv_data["leadership_activities"] = _apply_leadership_review(
            cv_data["leadership_activities"], reviews.get("leadership") or {}
        )

    skills_projects_review = reviews.get("skills_projects") or {}
    new_skills, new_projects = _apply_skills_projects_review(
        cv_data.get("skills") or [], cv_data.get("projects") or [], skills_projects_review
    )
    cv_data["skills"] = new_skills
    cv_data["projects"] = new_projects

    return cv_data


def render_reviewed_cv(cv: dict[str, Any], state: OrchestratorState, config: RunnableConfig) -> RenderedCv:
    """Generate the reviewed CV's PDF (step 7, and again after a chat edit)."""
    applicant_name = cv.get("name", "Applicant")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(applicant_name or "applicant"))[:80].strip("_").lower() or "applicant"
    pdf_name = f"{slug}_cv_reviewed.pdf"

    # The length the overall assessment recommended; the renderer keeps to it
    # where the content allows and says so where it doesn't.
    recommended = ((state.get("review_outputs") or {}).get("assessment") or {}).get("recommended_pages")
    files = list(state.get("generated_files") or [])
    pdf_msg = generate_cv_pdf.invoke({
        "cv_json": json.dumps(cv, default=str),
        "output_filename": pdf_name,
        "applicant_name": applicant_name,
        "output_dir": session_output_dir(config),
        "max_pages": recommended if recommended in (1, 2) else 0,
    })
    if isinstance(pdf_msg, str) and "Error" not in pdf_msg:
        filename = _extract_filename(pdf_msg) or pdf_name
        notes = [line for line in pdf_msg.splitlines() if line.startswith("Layout note:")]
        return RenderedCv(with_file(files, file_record(filename, "cv", config)), filename, notes, None)
    return RenderedCv(files, None, [], str(pdf_msg))


def cr_step7_finalize(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Assemble the revised CV and export a downloadable PDF via LaTeX."""
    revised_cv = _assemble_reviewed_cv(state)
    rendered = render_reviewed_cv(revised_cv, state, config)
    if rendered.filename:
        narration = "\n".join([
            "CV review complete. I've applied the approved suggestions and generated "
            f"a revised CV PDF: {rendered.filename}.",
            *rendered.notes,
        ])
    else:
        narration = f"CV review complete, but I ran into a problem generating the revised PDF: {rendered.error}"

    return _narrate(
        {"cv_data": revised_cv, "generated_files": rendered.files},
        narration,
    )


# Backwards-compat shim. Some older imports referenced cv_review_stub; keep
# the name resolving to the entry node so nothing breaks during the migration.
cv_review_stub = cr_entry
