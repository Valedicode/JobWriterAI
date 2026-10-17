"""
Job-tailoring branch nodes for the orchestrator graph.

Each step node is a deterministic wrapper around an existing ``@tool`` function
in ``app.agents.writer_agent`` or ``app.agents.scoring_agent``. The giant
prompt-driven sequencing that lived inside ``writer_agent.agent`` is now
expressed as the edges of the graph instead.

Gate nodes emit a ``GatePayload`` via ``langgraph.types.interrupt`` and route on
the user's ``GateResolution`` reply:

    [step]  generic step node:  call a tool, store output in state.
    [gate]  approval / choice:  interrupt(), then Command(goto=..., update=...).

The node names below match the labels in the architecture mermaid diagram.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, interrupt

from app.agents.orchestrator.gates import GatePayload, GateResolution
from app.agents.orchestrator.state import OrchestratorState
from app.agents.writer_agent import (
    decide_tailoring_strategy,
    generate_cover_letter_content,
    generate_cover_letter_pdf,
    generate_cv_pdf,
    generate_job_summary,
    rewrite_enhance_content,
    select_prioritize_content,
)
from app.agents.orchestrator.outputs import file_record, session_output_dir
from app.services.latex_renderer import cv_content_language, extract_recipient_name
from app.services.resume_adapter import normalize_skill_list
from app.agents.scoring_agent import (
    apply_eligibility_answers,
    calculate_compatibility_score_v2,
    eligibility_answers_from_text,
    eligibility_questions,
)
from app.agents.strategist_agent import plan_tailoring_strategy, validate_plan
from app.agents.translator_agent import needs_translation, translatable_fields, translate_cv, translate_cv_fields
from app.models.schemas import CompatibilityReport, StrategistPlan, VerifierResult
from app.services.bullet_index import index_cv_bullets as _index_cv_bullets
from app.services.fact_ledger import build_fact_ledger, eligibility_answer_facts, with_facts
from app.services.requirement_rubric import evidence_prompt_block, refresh_evidence, rubric_alignment
from app.services.verifier import issue_notes, retry_feedback, text_language, verify_rewrites


# --------------------------------------------------------------------------- #
# Node-name constants
# Keeping these as module-level constants so graph.py and the API layer can
# refer to them without duplicating string literals.
# --------------------------------------------------------------------------- #

JT_ENTRY = "jt_entry"
JT_STEP1_SUMMARIZE = "jt_step1_summarize_job"
JT_STEP2_COMPATIBILITY = "jt_step2_compute_compatibility"
JT_GATE_PRESENT_SCORE = "jt_gate_present_score"
JT_STEP3_STRATEGY = "jt_step3_decide_strategy"
JT_STEP4_SELECT = "jt_step4_select_prioritize"
JT_GATE_APPROVE_SELECTION = "jt_gate_approve_selection"
JT_STEP5_REWRITE = "jt_step5_rewrite_enhance"
JT_GATE_APPROVE_REWRITE = "jt_gate_approve_rewrite"
JT_STEP6_ASSEMBLE = "jt_step6_assemble_cv"
JT_STEP7_GENERATE_CV_FILES = "jt_step7_generate_cv_files"
JT_GATE_COVER_LETTER = "jt_gate_cover_letter"
JT_STEP_SET_COVER_LETTER_RECIPIENT = "jt_step_set_cover_letter_recipient"
JT_GATE_COVER_LETTER_RECIPIENT = "jt_gate_cover_letter_recipient"
JT_STEP8_COVER_LETTER_CONTENT = "jt_step8_cover_letter_content"
JT_GATE_APPROVE_COVER_LETTER = "jt_gate_approve_cover_letter"
JT_STEP9_COVER_LETTER_FILES = "jt_step9_cover_letter_files"
JT_GATE_STEP_FAILED = "jt_gate_step_failed"
JT_GATE_CONFIRM_ELIGIBILITY = "jt_gate_confirm_eligibility"
JT_GATE_CONFIRM_CHANGE = "jt_gate_confirm_change"
JT_STEP5B_VERIFY = "jt_step5b_verify"
JT_GATE_VERIFIER_FAILED = "jt_gate_verifier_failed"
JT_GATE_CV_LANGUAGE = "jt_gate_cv_language"
JT_STEP_TRANSLATE_CV = "jt_step_translate_cv"
JT_GATE_APPROVE_TRANSLATION = "jt_gate_approve_translation"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _narrate(state_update: dict[str, Any], text: str) -> dict[str, Any]:
    """
    Attach a narration string to a state update.

    The text becomes both ``last_narration`` (the next assistant chat bubble
    returned by the API) and a new ``AIMessage`` appended to ``messages`` via
    LangGraph's ``add_messages`` reducer.
    """
    state_update["last_narration"] = text
    state_update.setdefault("messages", []).append(AIMessage(content=text))
    state_update["pending_gate"] = None
    return state_update


def _safe_json_loads(text: str, fallback: Any) -> Any:
    """Defensive JSON parse for tool outputs that may have stringified errors."""
    if not isinstance(text, str):
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def _tool_error(payload: Any, tool_label: str) -> str | None:
    """
    The failure a writer tool reported, or None if it succeeded.

    Tools catch their own exceptions and return ``{"error": ...}`` next to
    empty result lists; output that is not a JSON object counts as a failure
    too. Either way the step must not treat the empty lists as a real answer.
    """
    if not isinstance(payload, dict):
        return f"the {tool_label} tool returned malformed output"
    error = payload.get("error")
    return str(error) if error else None


def _fail_step(step: str, label: str, detail: str) -> dict[str, Any]:
    """State update for a failed step: record it and say so plainly."""
    detail = detail[:300]
    return _narrate(
        {"step_error": {"step": step, "label": label, "detail": detail}},
        f"{label} failed: {detail}\n\nNothing in your CV was changed by this step. "
        "You can retry it or stop here.",
    )


def _slugify(name: str) -> str:
    """Filesystem-safe lowercase slug for output filenames."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "applicant")).strip("_").lower()
    return slug or "applicant"


def _filename_base(cv_data: dict[str, Any]) -> str:
    return _slugify(str(cv_data.get("name", "applicant")))


def _applicant_contact(cv_data: dict[str, Any]) -> str:
    email = str(cv_data.get("email", "") or "").strip()
    phone = str(cv_data.get("phone", "") or "").strip()
    if email and phone:
        return f"{email} | {phone}"
    return email or phone


def _required_skills(job_data: dict[str, Any] | None) -> list[str]:
    if not isinstance(job_data, dict):
        return []
    skills = job_data.get("required_skills") or []
    return [s for s in skills if isinstance(s, str)]


# --------------------------------------------------------------------------- #
# Bullet identity
#
# Every bullet-like string in the CV gets a stable id (built by
# app.services.bullet_index). The id survives selection -> rewrite -> merge so
# a rewrite is matched to its source bullet even if the model's echoed
# `original` text drifts slightly, and so the UI can address individual
# bullets for accept/keep-original/custom decisions.
# --------------------------------------------------------------------------- #

def _normalize_bullet_text(text: str) -> str:
    """Casefold, collapse whitespace, strip trailing punctuation - used as a
    fallback match key when an id round-trip fails or the model paraphrases
    the echoed original slightly."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.rstrip(".,;: ").casefold()


def _bullet_index_prompt_block(bullet_index: dict[str, dict[str, Any]]) -> str:
    """Render the id index as a numbered `[id] text` list for LLM prompts."""
    lines = [f"[{bullet_id}] {ref['text']}" for bullet_id, ref in bullet_index.items()]
    return "\n".join(lines) if lines else "(no bullets found)"


_FACTS_PROMPT_LIMIT = 250


def _facts_prompt_block(fact_ledger: dict[str, dict[str, Any]] | None) -> str:
    """The ledger's citable facts as `[fact_id] value` lines. Bullets are left
    out: the rewrite prompt already has them, with their ids."""
    lines = [
        f"[{fact_id}] {fact.get('value', '')}"
        for fact_id, fact in (fact_ledger or {}).items()
        if isinstance(fact, dict) and fact.get("kind") != "bullet"
    ]
    return "\n".join(lines[:_FACTS_PROMPT_LIMIT]) if lines else "(none)"


def _write_bullet_by_id(cv_data: dict[str, Any], bullet_id: str, new_text: str, bullet_index: dict[str, dict[str, Any]]) -> bool:
    """Write ``new_text`` into ``cv_data`` at the location named by ``bullet_id``."""
    ref = bullet_index.get(bullet_id)
    if ref is None:
        return False
    section, entry_idx, field, idx = ref["section"], ref["entry_idx"], ref["field"], ref["idx"]
    if section == "summary":
        cv_data["summary"] = new_text
        return True
    entries = cv_data.get(section)
    if not isinstance(entries, list) or entry_idx is None or entry_idx >= len(entries):
        return False
    entry = entries[entry_idx]
    if not isinstance(entry, dict):
        return False
    if idx is None:
        entry[field] = new_text
        return True
    values = entry.get(field)
    if not isinstance(values, list) or idx >= len(values):
        return False
    values[idx] = new_text
    return True


def _merge_rewrites_into_cv(
    cv_data: dict[str, Any],
    rewritten_bullets: list[dict[str, Any]],
    bullet_index: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Apply ``rewritten`` text in place of the source bullet.

    Matches by ``bullet_id`` against ``bullet_index`` first (robust to the
    model paraphrasing the echoed ``original``), then falls back to
    normalized-text matching against every bullet in the CV. If neither
    match succeeds the rewrite is dropped - we never fabricate a location -
    and its original text is returned in ``unplaced`` so the caller can warn.
    """
    merged = copy.deepcopy(cv_data)
    if not rewritten_bullets:
        return merged, []

    bullet_index = bullet_index or _index_cv_bullets(cv_data)
    by_normalized: dict[str, str] = {}
    for bullet_id, ref in bullet_index.items():
        by_normalized[_normalize_bullet_text(ref["text"])] = bullet_id

    unplaced: list[str] = []
    for entry in rewritten_bullets:
        if not isinstance(entry, dict):
            continue
        original = entry.get("original")
        rewritten = entry.get("rewritten")
        bullet_id = entry.get("bullet_id")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue

        target_id = bullet_id if isinstance(bullet_id, str) and bullet_id in bullet_index else None
        if target_id is None and isinstance(original, str):
            target_id = by_normalized.get(_normalize_bullet_text(original))

        if target_id is None or not _write_bullet_by_id(merged, target_id, rewritten, bullet_index):
            unplaced.append(original if isinstance(original, str) else rewritten)

    return merged, unplaced


def _without_bullets(
    cv_data: dict[str, Any], bullet_ids: list[str], bullet_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """A copy of the CV without the given bullets (the user chose to leave
    them out). List items are removed from the end, so the remaining indices
    stay right; a single text field (a project description, the summary) is
    emptied."""
    out = copy.deepcopy(cv_data)
    refs = [bullet_index[b] for b in bullet_ids if b in bullet_index]
    for ref in sorted(refs, key=lambda r: (r["idx"] is None, -(r["idx"] or 0))):
        if ref["section"] == "summary":
            out["summary"] = ""
            continue
        entries = out.get(ref["section"])
        if not isinstance(entries, list) or ref["entry_idx"] is None or ref["entry_idx"] >= len(entries):
            continue
        entry = entries[ref["entry_idx"]]
        if not isinstance(entry, dict):
            continue
        if ref["idx"] is None:
            entry[ref["field"]] = ""
        elif isinstance(entry.get(ref["field"]), list) and ref["idx"] < len(entry[ref["field"]]):
            del entry[ref["field"]][ref["idx"]]
    return out


def _extract_filename(tool_message: str) -> str | None:
    """Pull a filename out of a generate_* tool success message."""
    match = re.search(r"The file '([^']+)' is ready for download", tool_message or "")
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# Entry node
# --------------------------------------------------------------------------- #

def jt_entry(state: OrchestratorState) -> dict[str, Any]:
    """
    Validate inputs and produce the opening narration.

    We do NOT call the LLM here - the goal is to confirm the CV/job pair is
    present and well-formed before we burn tokens on step 1.
    """
    raw_cv = state.get("cv_data") or {}
    job_data = state.get("job_data") or {}
    name = raw_cv.get("name", "you")
    job_title = job_data.get("job_title") or "the target role"

    # Skills as plain names: extraction can leave the CV's category labels
    # inside the items ('Backend: FastAPI'), which would print as skills and
    # keep the plan's skill groups from matching.
    cv_data = {**raw_cv, "skills": normalize_skill_list(raw_cv.get("skills"))} if raw_cv.get("skills") else raw_cv

    narration = (
        f"Starting the job-tailoring flow for {name} against {job_title}. "
        "I'll summarise the posting, score the match, propose a tailoring "
        "strategy, then walk you through bullet selection and rewriting "
        "with explicit approvals before any documents are generated."
    )
    bullet_index = _index_cv_bullets(cv_data)
    return _narrate(
        {
            "cv_data": cv_data,
            "bullet_index": bullet_index,
            "original_cv_data": copy.deepcopy(raw_cv),
            "fact_ledger": build_fact_ledger(cv_data, bullet_index),
        },
        narration,
    )


# --------------------------------------------------------------------------- #
# Step 1: summarise job posting
# --------------------------------------------------------------------------- #

_CONDITION_LABELS = (
    ("work_mode", "Work mode"),
    ("location_constraint", "Location"),
    ("hours_per_week", "Hours/week"),
    ("start_date", "Start"),
    ("work_permit", "Work permit"),
)


def _job_conditions(job_data: dict[str, Any]) -> list[str]:
    """The knockout conditions the posting states, as 'Label: value'."""
    return [f"{label}: {job_data[key]}" for key, label in _CONDITION_LABELS
            if isinstance(job_data.get(key), str) and job_data[key].strip()]


def jt_step1_summarize_job(state: OrchestratorState) -> dict[str, Any]:
    job_data = state.get("job_data") or {}
    job_json = json.dumps(job_data)
    summary_json = generate_job_summary.invoke({"job_json": job_json})
    summary = _safe_json_loads(summary_json, {
        "role": job_data.get("job_title", "Unknown"),
        "responsibilities": job_data.get("responsibilities", []),
        "required_skills": _required_skills(job_data),
        "preferred_skills": job_data.get("preferred_skills", []),
        "key_notes": [],
    })

    narration_lines = [
        "Job summary",
        "",
        f"Role: {summary.get('role', '')}",
    ]
    resp = summary.get("responsibilities") or []
    if resp:
        narration_lines.append("Responsibilities:")
        narration_lines.extend(resp[:6])
    req = summary.get("required_skills") or []
    if req:
        narration_lines.append("Required skills: " + ", ".join(req[:12]))
    pref = summary.get("preferred_skills") or []
    if pref:
        narration_lines.append("Preferred skills: " + ", ".join(pref[:8]))
    learnable = job_data.get("learnable_skills") or []
    if learnable:
        narration_lines.append("Taught on the job: " + ", ".join(learnable[:8]))
    conditions = _job_conditions(job_data)
    if conditions:
        narration_lines.append("Conditions: " + " | ".join(conditions))
    notes = summary.get("key_notes") or []
    if notes:
        narration_lines.append("Notes: " + " | ".join(notes[:4]))

    return _narrate({"job_summary": summary}, "\n".join(narration_lines))


# --------------------------------------------------------------------------- #
# Step 2: compute compatibility score (v2)
# --------------------------------------------------------------------------- #

def jt_step2_compute_compatibility(state: OrchestratorState) -> dict[str, Any]:
    cv_json = json.dumps(state.get("cv_data") or {})
    job_json = json.dumps(state.get("job_data") or {})
    report_json = calculate_compatibility_score_v2.invoke({
        "cv_json": cv_json,
        "job_json": job_json,
    })
    report = _safe_json_loads(report_json, None)
    error = _tool_error(report, "scoring") or _scoring_failure(report)
    if error:
        return _fail_step(JT_STEP2_COMPATIBILITY, "Compatibility scoring", error)

    score = report.get("aggregate_score", 0.0)
    level = (report.get("level") or "unknown").upper()
    lines = [
        "Compatibility analysis",
        "",
        f"Compatibility Score: {score:.2f} ({level})",
        "",
        "Dimension Breakdown:",
    ]
    for dim in report.get("dimensions", []) or []:
        name = dim.get("name", "?")
        ds = dim.get("score", 0.0)
        dw = dim.get("weight", 0.0)
        lines.append(f"{name} ({int(round(dw * 100))}%): {ds:.2f}")
    gap = report.get("gap_analysis", {}) or {}
    matched = [m for m in (gap.get("matched_skills") or []) if isinstance(m, dict)]
    transferable = [m for m in (gap.get("transferable_skills") or []) if isinstance(m, dict)]
    learnable = [m for m in (gap.get("learnable_skills") or []) if isinstance(m, dict)]
    missing = [m for m in (gap.get("missing_skills") or []) if isinstance(m, dict)]
    lines.append("")
    lines.append(
        f"Matched: {len(matched)} | Transferable: {len(transferable)} | "
        f"Learnable: {len(learnable)} | Missing: {len(missing)}"
    )
    # Name the actual gaps instead of only a count, and keep must-haves apart
    # from nice-to-haves: a missing required skill is a real gap, a missing
    # preferred one on a student posting is expected.
    missing_required = [m["required_skill"] for m in missing if m.get("required_skill") and m.get("is_required", True)]
    missing_preferred = [m["required_skill"] for m in missing if m.get("required_skill") and not m.get("is_required", True)]
    if missing_required:
        lines.append(f"Missing required skills: {', '.join(missing_required)}")
    if missing_preferred:
        lines.append(f"Missing nice-to-haves: {', '.join(missing_preferred)}")
    def _via(items: list[dict[str, Any]]) -> list[str]:
        return [
            f"{m['required_skill']} (via {m['matched_with']})" if m.get("matched_with") else m["required_skill"]
            for m in items
            if m.get("required_skill")
        ]

    if transferable:
        lines.append(f"Covered via a closely related skill: {', '.join(_via(transferable))}")
    if learnable:
        lines.append(f"Not on your CV, but learnable from adjacent experience: {', '.join(_via(learnable))}")

    eligibility = [e for e in (gap.get("eligibility") or []) if isinstance(e, dict)]
    if eligibility:
        lines.append("")
        lines.append("Eligibility (languages, degree, enrollment):")
        status_label = {"met": "met", "unclear": "unclear", "unmet": "NOT met"}
        for e in eligibility:
            label = status_label.get(e.get("status"), e.get("status", "?"))
            rationale = e.get("rationale") or ""
            lines.append(f"- {e.get('requirement')}: {label}" + (f" - {rationale}" if rationale else ""))

    for w in report.get("warnings") or []:
        if isinstance(w, str) and w:
            lines.append("")
            lines.append(w)
    if report.get("interpretation"):
        lines.append("")
        lines.append(report["interpretation"])
    gaps_required = missing_required + [
        m["required_skill"] for m in learnable if m.get("required_skill") and m.get("is_required", True)
    ]
    if gaps_required:
        lines.append(
            "\nNote: the rewrite step only rephrases what's already on your CV; it will not add "
            f"{', '.join(gaps_required)}. If you have any real exposure to them (a course, a side "
            "project, brief use on the job), add it to your CV before this step so it can be surfaced."
        )
    unclear = [e.get("requirement") for e in eligibility if e.get("status") == "unclear"]
    if unclear:
        follow_up = (
            "I'll ask you about these next."
            if _eligibility_questions_for(report) else
            "If you meet them, add them to the CV (a Languages section, an 'Expected' graduation date)."
        )
        lines.append(
            "\nYour CV gives no information on: "
            f"{', '.join(str(u) for u in unclear if u)}. These are left out of the score until "
            f"confirmed. {follow_up}"
        )

    return _narrate({"compatibility_report": report, "step_error": None}, "\n".join(lines))


def _scoring_failure(report: dict[str, Any]) -> str | None:
    """The scoring tool catches its own errors and returns an empty report
    (level 'unknown', no dimensions); that is a failed step, not a 0.0 score."""
    if report.get("level") == "unknown" and not report.get("dimensions"):
        return str(report.get("interpretation") or "the scoring returned no result")
    return None


# --------------------------------------------------------------------------- #
# Gate: confirm eligibility (only when the CV is silent on a stated must-have)
# --------------------------------------------------------------------------- #

def _eligibility_questions_for(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    checks = ((report or {}).get("gap_analysis") or {}).get("eligibility") or []
    try:
        return eligibility_questions(checks)
    except Exception:
        return []


def route_after_score(state: OrchestratorState) -> str:
    if state.get("step_error"):
        return JT_GATE_STEP_FAILED
    if not state.get("eligibility_asked") and _eligibility_questions_for(state.get("compatibility_report")):
        return JT_GATE_CONFIRM_ELIGIBILITY
    return _after_scoring(state)


def jt_gate_confirm_eligibility(state: OrchestratorState) -> Command:
    """
    Ask about language / enrollment / degree requirements the CV says
    nothing about, then rescore deterministically (no new LLM call) and
    continue to the strategy and the score review. Asked once per run; skipping leaves the
    items unclear and out of the score.
    """
    report = state.get("compatibility_report") or {}
    questions = _eligibility_questions_for(report)
    payload = GatePayload(
        step="confirm_eligibility",
        kind="input",
        narration=(
            "The posting asks for things your CV doesn't mention. Answer what applies; "
            "anything you skip stays out of the score."
        ),
        preview={"questions": questions},
        allowed_actions=["edit", "approve"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    answers = {k: v for k, v in (resolution.answers or {}).items() if isinstance(v, str) and v.strip()}
    if resolution.action == "edit" and not answers:
        # Typed through the chat rather than the form: read what we can.
        answers = eligibility_answers_from_text(questions, resolution.feedback or "")
        if not answers:
            return Command(
                update=_narrate({}, "I couldn't match that to the questions. Please answer them in the form, or skip."),
                goto=JT_GATE_CONFIRM_ELIGIBILITY,
            )
    if resolution.action != "edit" or not answers:
        return Command(
            update=_narrate(
                {"eligibility_asked": True},
                "Skipped. Those requirements stay out of the score.",
            ),
            goto=_after_scoring(state),
        )

    previous = CompatibilityReport.model_validate(report)
    cv_data, rescored, notes = apply_eligibility_answers(
        state.get("cv_data") or {}, state.get("job_data") or {}, previous, answers
    )
    asked = {q["requirement"] for q in questions}
    status_label = {"met": "met", "unclear": "still unclear", "unmet": "NOT met"}
    lines = [
        "Eligibility updated",
        "",
        f"Compatibility Score: {previous.aggregate_score:.2f} -> {rescored.aggregate_score:.2f} "
        f"({rescored.level.upper()})",
    ]
    for check in rescored.gap_analysis.eligibility:
        if check.requirement in asked:
            lines.append(f"- {check.requirement}: {status_label.get(check.status, check.status)}")
    lines.extend(f"\n{w}" for w in rescored.warnings)
    if notes:
        lines.append("")
        lines.extend(notes)
    return Command(
        update=_narrate(
            {
                "cv_data": cv_data,
                "fact_ledger": with_facts(state.get("fact_ledger"), eligibility_answer_facts(questions, answers)),
                "compatibility_report": rescored.model_dump(),
                "eligibility_asked": True,
            },
            "\n".join(lines),
        ),
        goto=_after_scoring(state),
    )


# --------------------------------------------------------------------------- #
# Gate: present score and plan. The fit and the strategist's plan are approved
# together; edit redrafts the plan with the user's notes.
# --------------------------------------------------------------------------- #

def _plan_preview(state: OrchestratorState) -> dict[str, Any] | None:
    """The validated plan in reviewable form: featured bullets as their text,
    keywords split into usable and not on the CV."""
    strategy = state.get("tailoring_strategy") or {}
    plan = strategy.get("plan")
    if not isinstance(plan, dict):
        return None
    bullet_index = state.get("bullet_index") or {}
    keywords = [k for k in plan.get("keyword_plan") or [] if isinstance(k, dict)]
    return {
        "positioning": plan.get("positioning", ""),
        "headline": plan.get("headline"),
        "cv_language": plan.get("cv_language"),
        "convention": plan.get("convention"),
        "section_order": plan.get("section_order") or [],
        "featured_bullets": [
            {"bullet_id": b, "text": (bullet_index.get(b) or {}).get("text", "")}
            for b in plan.get("bullets_to_feature") or []
        ],
        "skill_groups": [
            {"label": g.get("label", ""), "skills": g.get("skills", [])}
            for g in plan.get("skill_groups") or [] if isinstance(g, dict)
        ],
        "keywords_to_use": [k["keyword"] for k in keywords if k.get("allowed")],
        "keywords_not_on_cv": [k["keyword"] for k in keywords if not k.get("allowed")],
        "red_flags": plan.get("red_flags") or [],
        "notes": strategy.get("plan_notes") or [],
    }


def jt_gate_present_score(state: OrchestratorState) -> Command:
    report = state.get("compatibility_report") or {}
    plan = _plan_preview(state)
    payload = GatePayload(
        step="present_score",
        kind="approval",
        narration=(
            "Review the fit and the tailoring plan. Approve to start selecting bullets, "
            "edit with notes to redraft the plan, or reject to stop."
            if plan else
            "Review the fit. Approve to continue with the rule-based strategy, edit with notes "
            "to try drafting the plan again, or reject to stop."
        ),
        preview={
            "aggregate_score": report.get("aggregate_score"),
            "level": report.get("level"),
            "dimensions": report.get("dimensions", []),
            "gap_analysis": report.get("gap_analysis", {}),
            "interpretation": report.get("interpretation", ""),
            "warnings": report.get("warnings", []),
            "rubric": report.get("rubric", []),
            "knockouts": report.get("knockouts", []),
            "plan": plan,
        },
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        return Command(
            update={"pending_gate": None},
            goto=JT_STEP4_SELECT,
        )
    if resolution.action == "edit":
        notes = list(state.get("strategy_feedback") or []) + [resolution.feedback or ""]
        return Command(
            update=_narrate({"strategy_feedback": notes},
                            f"Redrafting the plan with your notes: {resolution.feedback}"),
            goto=JT_STEP3_STRATEGY,
        )
    return Command(
        update=_narrate({}, "Stopped at the compatibility-score review. No documents were generated."),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Step 3: decide tailoring strategy
# --------------------------------------------------------------------------- #

def jt_step3_decide_strategy(state: OrchestratorState) -> dict[str, Any]:
    compatibility_json = json.dumps(state.get("compatibility_report") or {})
    cv_json = json.dumps(state.get("cv_data") or {})
    job_json = json.dumps(state.get("job_data") or {})
    strategy_json = decide_tailoring_strategy.invoke({
        "compatibility_score_json": compatibility_json,
        "cv_json": cv_json,
        "job_json": job_json,
    })
    strategy = _safe_json_loads(strategy_json, {
        "strategy": "standard",
        "intensity": "moderate",
        "focus_areas": [],
        "directives": [],
    })

    # The rule-based directives stay underneath (they carry the anti-
    # fabrication rules); the strategist's plan adds the judgement calls.
    plan, plan_notes, plan_error = _draft_plan(state)
    strategy["plan"] = plan
    strategy["plan_notes"] = plan_notes

    focus = ", ".join(strategy.get("focus_areas", []) or []) or "none"
    lines = [
        "Tailoring strategy",
        "",
        f"Strategy: {strategy.get('strategy', '')}",
        f"Intensity: {strategy.get('intensity', 'moderate')}",
        f"Focus areas: {focus}",
        f"Directives: {len(strategy.get('directives', []) or [])}",
    ]
    if plan:
        lines += ["", f"Positioning: {plan.get('positioning', '')}"]
        if plan.get("headline"):
            lines.append(f"Headline: {plan['headline']}")
        lines.append(f"Featured bullets: {len(plan.get('bullets_to_feature') or [])}")
    else:
        lines += ["", f"I couldn't draft the tailoring plan ({plan_error}). The rule-based strategy above still applies."]
    return _narrate({"tailoring_strategy": strategy}, "\n".join(lines))


def _draft_plan(state: OrchestratorState) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """(validated plan, notes on what validation changed, error). The model's
    plan is only advice until validate_plan has checked it against the CV."""
    cv_data = state.get("cv_data") or {}
    report = state.get("compatibility_report") or {}
    bullet_index = state.get("bullet_index") or _index_cv_bullets(cv_data)
    ledger = state.get("fact_ledger") or {}
    feedback = "\n".join(f"- {note}" for note in state.get("strategy_feedback") or [] if note)

    raw = _safe_json_loads(plan_tailoring_strategy.invoke({
        "cv_json": json.dumps(cv_data),
        "job_json": json.dumps(state.get("job_data") or {}),
        "report_json": json.dumps(report),
        "facts_block": _facts_prompt_block(ledger),
        "bullet_index_block": _bullet_index_prompt_block(bullet_index),
        "feedback": feedback,
        "cv_language": state.get("cv_language") or "",
    }), None)
    error = _tool_error(raw, "strategist")
    if error:
        return None, [], error[:200]
    try:
        plan = StrategistPlan.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - a malformed plan is reported, not raised
        return None, [], f"the plan was malformed: {exc}"[:200]
    projects = cv_data.get("projects") if isinstance(cv_data.get("projects"), list) else []
    skills = cv_data.get("skills") if isinstance(cv_data.get("skills"), list) else []
    cleaned, notes = validate_plan(
        plan, rubric=report.get("rubric") or [], ledger=ledger,
        bullet_index=bullet_index, project_count=len(projects),
        cv_skills=[s for s in skills if isinstance(s, str)],
    )
    if state.get("cv_language") in ("english", "german"):
        # The user decided the CV language; the plan doesn't get to change it.
        cleaned = cleaned.model_copy(update={"cv_language": state["cv_language"]})
    return cleaned.model_dump(), notes, None


# --------------------------------------------------------------------------- #
# Step 4: select & prioritise content
# --------------------------------------------------------------------------- #

def jt_step4_select_prioritize(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    job_json = json.dumps(state.get("job_data") or {})
    strategy_json = json.dumps(state.get("tailoring_strategy") or {})
    bullet_index = state.get("bullet_index") or _index_cv_bullets(cv_data)

    selected_json = select_prioritize_content.invoke({
        "cv_json": json.dumps(cv_data),
        "job_json": job_json,
        "tailoring_strategy_json": strategy_json,
        "bullet_index_block": _bullet_index_prompt_block(bullet_index),
        "evidence_block": evidence_prompt_block((state.get("compatibility_report") or {}).get("rubric") or []),
    })
    selected = _safe_json_loads(selected_json, None)
    error = _tool_error(selected, "selection")
    if error:
        return _fail_step(JT_STEP4_SELECT, "Bullet selection", error)

    # Drop bullets the model matched to an id that doesn't exist - never
    # trust an echoed id blindly (guards against hallucinated ids).
    bullets = selected.get("selected_bullets", []) or []
    for b in bullets:
        if isinstance(b, dict) and isinstance(b.get("bullet_id"), str) and b["bullet_id"] not in bullet_index:
            b["bullet_id"] = None

    fallback_used = False
    if not bullets and bullet_index:
        # The model answered but selected nothing, despite the CV having
        # content - a low-relevance CV/job pairing is not a reason to skip
        # tailoring entirely. Fall back to the CV's first bullets so rewrite
        # still has something real to work on, flagged with an honest low
        # relevance score rather than a fabricated one. (A failed tool call
        # never gets here; it goes to the step_failed gate above.)
        fallback_used = True
        bullets = [
            {
                "section": ref["section"],
                "original_text": ref["text"],
                "relevance_score": 0.1,
                "reason": "Selection step returned nothing usable; included as a best-available fallback.",
                "bullet_id": bullet_id,
            }
            for bullet_id, ref in list(bullet_index.items())[:5]
        ]
    rubric = (state.get("compatibility_report") or {}).get("rubric") or []
    selected["selected_bullets"] = [
        b if fallback_used or not isinstance(b, dict) else _with_match_labels(b, rubric) for b in bullets
    ]
    bullets = selected["selected_bullets"]

    order = selected.get("section_order", []) or []
    lines = [
        "Selected content",
        "",
        f"Top bullets: {len(bullets)}",
        f"Recommended section order: {', '.join(order) if order else '(unchanged)'}",
        f"Emphasised sections: {', '.join(selected.get('sections_to_emphasize', []) or []) or '(none)'}",
    ]
    if fallback_used:
        lines.append(
            f"\nNote: the selection step picked no bullets, so I included the first {len(bullets)} "
            "bullets of your CV in their original order. Exclude any you don't want below."
        )
    return _narrate(
        {"selected_content": selected, "bullet_index": bullet_index, "step_error": None},
        "\n".join(lines),
    )


# Match labels for the selection review. The relevance score is the selection
# model's own estimate, so users see a word instead of the number, and the
# deterministic rubric decides whether a bullet shows a requirement.
STRONG_MATCH = 0.7
# Below this, a bullet that shows no requirement of the posting is recommended
# for leaving out of the tailored CV (never pre-selected).
LEAVE_OUT_BELOW = 0.35


def _with_match_labels(bullet: dict[str, Any], rubric: list[dict[str, Any]]) -> dict[str, Any]:
    """The bullet with `match` (strong / partial / weak), `shows` (the posting
    requirements it evidences, from the rubric) and `leave_out_recommended`."""
    try:
        relevance = float(bullet.get("relevance_score"))
    except (TypeError, ValueError):
        relevance = None
    bullet_id = bullet.get("bullet_id")
    shows = [
        row.get("text", "") for row in rubric
        if isinstance(row, dict) and row.get("category") in ("must_have", "nice_to_have")
        and bullet_id and bullet_id in (row.get("evidence_bullet_ids") or [])
    ]
    if relevance is None:
        match = None
    elif relevance >= STRONG_MATCH:
        match = "strong"
    elif relevance >= LEAVE_OUT_BELOW:
        match = "partial"
    else:
        match = "weak"
    return {
        **bullet,
        "match": match,
        "shows": shows,
        "leave_out_recommended": match == "weak" and not shows,
    }


# --------------------------------------------------------------------------- #
# Gate: approve selection (HARD GATE - edit loops back to step 4)
# --------------------------------------------------------------------------- #

def _finalize_selection_decisions(
    selected_bullets: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Apply per-bullet decisions: include (tailor it, the default), exclude
    (keep it as it is: not rewritten, still on the CV) or leave_out (drop it
    from the tailored CV). Returns the bullets to rewrite and the ids left out.
    """
    decisions = decisions or {}
    kept: list[dict[str, Any]] = []
    left_out: list[str] = []
    for b in selected_bullets:
        if not isinstance(b, dict):
            continue
        bullet_id = b.get("bullet_id")
        decision = decisions.get(bullet_id) if bullet_id else None
        choice = getattr(decision, "choice", None) if decision else "include"
        if choice == "leave_out" and bullet_id:
            left_out.append(bullet_id)
            continue
        if choice == "exclude":
            continue
        kept.append(b)
    return kept, left_out


def jt_gate_approve_selection(state: OrchestratorState) -> Command:
    selected = state.get("selected_content") or {}
    payload = GatePayload(
        step="approve_selection",
        kind="approval",
        narration=(
            "Review the selected bullets and proposed section ordering. "
            "Approve to continue to rewriting, or edit with feedback to "
            "re-run selection."
        ),
        preview=selected,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        selected_bullets = selected.get("selected_bullets", []) or []
        finalized, left_out = _finalize_selection_decisions(selected_bullets, resolution.decisions)
        kept_as_is = len(selected_bullets) - len(finalized) - len(left_out)
        updated_selected = dict(selected)
        updated_selected["selected_bullets"] = finalized
        update: dict[str, Any] = {
            "pending_gate": None,
            "selected_content": updated_selected,
            "left_out_bullet_ids": left_out,
        }
        if kept_as_is or left_out:
            parts = [f"Tailoring {len(finalized)} bullet(s)"]
            if kept_as_is:
                parts.append(f"{kept_as_is} kept as they are")
            if left_out:
                parts.append(f"{len(left_out)} left out of the tailored CV")
            update = _narrate(update, ", ".join(parts) + ".")
        return Command(update=update, goto=JT_STEP5_REWRITE)
    if resolution.action == "edit":
        # Merge feedback into strategy so the next pass of step 4 sees it.
        strategy = dict(state.get("tailoring_strategy") or {})
        feedback_history = list(strategy.get("user_feedback", []) or [])
        feedback_history.append({"stage": "approve_selection", "text": resolution.feedback or ""})
        strategy["user_feedback"] = feedback_history
        return Command(
            update=_narrate(
                {"tailoring_strategy": strategy, "selected_content": None},
                f"Got it - re-running selection with your feedback: {resolution.feedback}",
            ),
            goto=JT_STEP4_SELECT,
        )
    return Command(
        update=_narrate({}, "Stopped before rewriting. No documents were generated."),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Step 5: rewrite & enhance content
# --------------------------------------------------------------------------- #

def jt_step5_rewrite_enhance(state: OrchestratorState) -> dict[str, Any]:
    cv_json = json.dumps(state.get("cv_data") or {})
    job_json = json.dumps(state.get("job_data") or {})
    selected_json = json.dumps(state.get("selected_content") or {})
    strategy_json = json.dumps(state.get("tailoring_strategy") or {})
    locked = state.get("locked_rewrites") or []

    rewritten_json = rewrite_enhance_content.invoke({
        "cv_json": cv_json,
        "job_json": job_json,
        "selected_content_json": selected_json,
        "tailoring_strategy_json": strategy_json,
        "locked_bullets_json": json.dumps(locked),
        "facts_block": _facts_prompt_block(state.get("fact_ledger")),
    })
    rewritten = _safe_json_loads(rewritten_json, None)
    error = _tool_error(rewritten, "rewrite")
    if error:
        # locked_rewrites stays in state so a retry keeps the user's choices.
        return _fail_step(JT_STEP5_REWRITE, "Bullet rewriting", error)
    rewritten["user_written_bullet_ids"] = [
        b["bullet_id"] for b in locked
        if isinstance(b, dict) and b.get("user_written") and b.get("bullet_id")
    ]

    bullets = rewritten.get("rewritten_bullets", []) or []
    selected_bullets = (state.get("selected_content") or {}).get("selected_bullets", []) or []
    fallback_used = False
    if not bullets and selected_bullets:
        # The rewrite model answered but returned no rewrites despite being
        # given bullets to work with. Echo the selections back unchanged
        # rather than silently producing a no-op tailoring pass - the user can
        # still see and accept/edit them individually at the gate. (A failed
        # tool call never gets here; it goes to the step_failed gate above.)
        fallback_used = True
        bullets = [
            {
                "original": sb.get("original_text", ""),
                "rewritten": sb.get("original_text", ""),
                "confidence": 0.0,
                "keywords_added": [],
                "bullet_id": sb.get("bullet_id"),
            }
            for sb in selected_bullets
            if isinstance(sb, dict) and sb.get("original_text")
        ]
        rewritten["rewritten_bullets"] = bullets

    avg_conf = (
        sum(float(b.get("confidence", 0.0)) for b in bullets) / max(1, len(bullets))
        if bullets else 0.0
    )
    if fallback_used:
        lines = [
            "Rewritten content",
            "",
            f"Bullets rewritten: 0 ({len(bullets)} shown unchanged)",
            "\nNote: the rewrite step returned no rewrites, so the selected bullets are shown "
            "unchanged below - edit them at the gate if you'd like different wording.",
        ]
    else:
        lines = [
            "Rewritten content",
            "",
            f"Bullets rewritten: {len(bullets)}",
            f"Average rewrite confidence: {avg_conf:.2f}",
            f"Keywords woven in: {', '.join(rewritten.get('keywords_inserted', []) or []) or '(none)'}",
        ]
    return _narrate(
        {"rewritten_content": rewritten, "locked_rewrites": None, "step_error": None},
        "\n".join(lines),
    )


# --------------------------------------------------------------------------- #
# Step 5b: fact-check the rewrites against the ledger (no LLM)
#
# A rewrite that claims something no fact supports (a technology, number or
# year) is sent back once with the problems as feedback, the passing bullets
# locked. Still blocked after that, the verifier_failed gate shows the
# problems; a blocked rewrite never reaches the review gate as-is.
# --------------------------------------------------------------------------- #

MAX_VERIFY_ATTEMPTS = 2

_REWRITE_ANNOTATIONS = ("verifier_notes", "user_written")


def _without_annotations(bullet: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in bullet.items() if k not in _REWRITE_ANNOTATIONS}


def _annotated(rewritten: dict[str, Any], result: VerifierResult) -> dict[str, Any]:
    """The rewrites with each bullet's verifier notes attached for the UI."""
    notes = issue_notes(result)
    annotated = dict(rewritten)
    annotated["rewritten_bullets"] = [
        {**_without_annotations(b), "verifier_notes": notes.get(b.get("bullet_id") or "", [])}
        for b in rewritten.get("rewritten_bullets") or [] if isinstance(b, dict)
    ]
    annotated["summary_notes"] = notes.get("summary", [])
    return annotated


def _blocked_problem_lines(result: VerifierResult) -> list[str]:
    return [f"{i.bullet_id}: {i.detail}" for i in result.issues if i.severity == "block"]


def jt_step5b_verify(state: OrchestratorState) -> Command:
    rewritten = state.get("rewritten_content") or {}
    attempt = state.get("verify_attempt") or 1
    bullet_index = state.get("bullet_index") or _index_cv_bullets(state.get("cv_data") or {})
    result = verify_rewrites(
        rewritten,
        state.get("fact_ledger") or {},
        bullet_index,
        cv_summary=(state.get("cv_data") or {}).get("summary"),
        attempt=attempt,
        user_written_ids=rewritten.get("user_written_bullet_ids") or [],
    )
    warnings = sum(1 for i in result.issues if i.severity == "warn")

    if result.passed:
        narration = "Fact check: every rewrite is backed by your CV."
        if warnings:
            narration += f" {warnings} thing(s) are worth a second look; they're marked below."
        return Command(
            update=_narrate(
                {"rewritten_content": _annotated(rewritten, result),
                 "verifier_result": result.model_dump(), "verify_attempt": None},
                narration,
            ),
            goto=JT_GATE_APPROVE_REWRITE,
        )

    blocked = set(result.blocked_ids())
    problems = _blocked_problem_lines(result)
    if attempt < MAX_VERIFY_ATTEMPTS:
        user_written = set(rewritten.get("user_written_bullet_ids") or [])
        locked = [
            {**_without_annotations(b), **({"user_written": True} if b.get("bullet_id") in user_written else {})}
            for b in rewritten.get("rewritten_bullets") or []
            if isinstance(b, dict) and b.get("bullet_id") and b["bullet_id"] not in blocked
        ]
        strategy = dict(state.get("tailoring_strategy") or {})
        strategy["user_feedback"] = list(strategy.get("user_feedback") or []) + [
            {"stage": "verifier", "text": retry_feedback(result)}
        ]
        return Command(
            update=_narrate(
                {"locked_rewrites": locked, "tailoring_strategy": strategy, "rewritten_content": None,
                 "verifier_result": result.model_dump(), "verify_attempt": attempt + 1},
                "Fact check: some rewrites claim things your CV doesn't support:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nRewriting those again; the others are kept.",
            ),
            goto=JT_STEP5_REWRITE,
        )

    return Command(
        update=_narrate(
            {"rewritten_content": _annotated(rewritten, result),
             "verifier_result": result.model_dump(), "verify_attempt": None},
            "Fact check: these rewrites still claim things your CV doesn't support:\n"
            + "\n".join(f"- {p}" for p in problems),
        ),
        goto=JT_GATE_VERIFIER_FAILED,
    )


def jt_gate_verifier_failed(state: OrchestratorState) -> Command:
    """
    approve: keep the original wording of every blocked bullet (and the CV's
             own summary if the new one was blocked), then review as usual;
    edit:    re-run the rewrite with the user's notes, passing bullets kept;
    reject:  stop.
    A blocked rewrite is never applied from here.
    """
    rewritten = state.get("rewritten_content") or {}
    result = VerifierResult.model_validate(state.get("verifier_result") or {"passed": False, "attempt": 1,
                                                                            "preservation_score": 1.0})
    blocked = set(result.blocked_ids())
    bullets = [b for b in rewritten.get("rewritten_bullets") or [] if isinstance(b, dict)]
    payload = GatePayload(
        step="verifier_failed",
        kind="approval",
        narration=(
            "Some rewrites claim things your CV doesn't support. Approve to keep your original "
            "wording for those, edit to rewrite them with your notes, or reject to stop."
        ),
        preview={
            "problems": _blocked_problem_lines(result),
            "blocked_rewrites": [
                {"bullet_id": b.get("bullet_id"), "original": b.get("original", ""), "rewritten": b.get("rewritten", "")}
                for b in bullets if b.get("bullet_id") in blocked
            ],
            "on_approve": "Blocked bullets keep their original wording; the other rewrites go to your review.",
        },
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        kept = [
            {**b, "rewritten": b.get("original", ""), "keywords_added": [], "cited_fact_ids": [],
             "verifier_notes": ["Kept your original wording: the rewrite wasn't backed by your CV."]}
            if b.get("bullet_id") in blocked else b
            for b in bullets
        ]
        updated = {**rewritten, "rewritten_bullets": kept}
        if "summary" in blocked:
            updated["updated_summary"] = ""
            updated["summary_notes"] = ["Kept your CV's own summary: the new one wasn't backed by your CV."]
        return Command(
            update=_narrate({"rewritten_content": updated},
                            f"Kept the original wording for {len(blocked)} item(s). Review the rest below."),
            goto=JT_GATE_APPROVE_REWRITE,
        )
    if resolution.action == "edit":
        strategy = dict(state.get("tailoring_strategy") or {})
        strategy["user_feedback"] = list(strategy.get("user_feedback") or []) + [
            {"stage": "approve_rewrite", "text": resolution.feedback or ""}
        ]
        user_written = set(rewritten.get("user_written_bullet_ids") or [])
        locked = [
            {**_without_annotations(b), **({"user_written": True} if b.get("bullet_id") in user_written else {})}
            for b in bullets if b.get("bullet_id") and b["bullet_id"] not in blocked
        ]
        return Command(
            update=_narrate(
                {"tailoring_strategy": strategy, "rewritten_content": None, "locked_rewrites": locked,
                 "verify_attempt": None},
                f"Re-running the rejected rewrites with your notes: {resolution.feedback}",
            ),
            goto=JT_STEP5_REWRITE,
        )
    return Command(
        update=_narrate({}, "Stopped before document assembly. No documents were generated."),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Gate: approve rewrite (HARD GATE - edit loops back to step 5)
# --------------------------------------------------------------------------- #

def _finalize_rewrite_decisions(
    rewritten_bullets: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Apply per-bullet accept/keep_original/custom decisions. Default: accept,
    so a plain Approve with no decisions keeps today's behaviour of applying
    every rewrite as-is.
    """
    decisions = decisions or {}
    out: list[dict[str, Any]] = []
    for b in rewritten_bullets:
        if not isinstance(b, dict):
            continue
        bullet_id = b.get("bullet_id")
        decision = decisions.get(bullet_id) if bullet_id else None
        choice = getattr(decision, "choice", None) if decision else "accept"
        if choice == "keep_original":
            nb = dict(b)
            nb["rewritten"] = b.get("original", "")
            nb["keywords_added"] = []
            out.append(nb)
        elif choice == "custom" and getattr(decision, "text", None):
            nb = dict(b)
            nb["rewritten"] = decision.text
            out.append(nb)
        else:
            out.append(b)
    return out


def _locked_bullets_for_edit(
    rewritten_bullets: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    For the edit path: only bullets with an EXPLICIT decision are locked and
    echoed unchanged into the next rewrite pass. Bullets with no decision are
    considered still in play and get regenerated with the new feedback.
    """
    decisions = decisions or {}
    locked: list[dict[str, Any]] = []
    for b in rewritten_bullets:
        if not isinstance(b, dict):
            continue
        b = _without_annotations(b)
        bullet_id = b.get("bullet_id")
        decision = decisions.get(bullet_id) if bullet_id else None
        if decision is None:
            continue
        choice = getattr(decision, "choice", None)
        if choice == "keep_original":
            locked.append({**b, "rewritten": b.get("original", ""), "keywords_added": []})
        elif choice == "custom" and getattr(decision, "text", None):
            # The user's own words: the fact check leaves them alone.
            locked.append({**b, "rewritten": decision.text, "user_written": True})
        elif choice == "accept":
            locked.append(b)
    return locked


def jt_gate_approve_rewrite(state: OrchestratorState) -> Command:
    rewritten = state.get("rewritten_content") or {}
    payload = GatePayload(
        step="approve_rewrite",
        kind="approval",
        narration=(
            "Review each rewritten bullet against its original. Approve to "
            "assemble the tailored CV, edit with notes to re-run the rewrite, "
            "or reject to stop."
        ),
        preview=rewritten,
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        bullets = rewritten.get("rewritten_bullets", []) or []
        finalized = _finalize_rewrite_decisions(bullets, resolution.decisions)
        rejected_count = len(bullets) - len(finalized)
        updated_rewritten = dict(rewritten)
        updated_rewritten["rewritten_bullets"] = finalized
        update: dict[str, Any] = {"pending_gate": None, "rewritten_content": updated_rewritten}
        if rejected_count:
            update = _narrate(update, f"Applying {len(finalized)} rewrites ({rejected_count} reverted to original).")
        return Command(update=update, goto=JT_STEP6_ASSEMBLE)
    if resolution.action == "edit":
        strategy = dict(state.get("tailoring_strategy") or {})
        feedback_history = list(strategy.get("user_feedback", []) or [])
        feedback_history.append({"stage": "approve_rewrite", "text": resolution.feedback or ""})
        strategy["user_feedback"] = feedback_history
        bullets = rewritten.get("rewritten_bullets", []) or []
        locked = _locked_bullets_for_edit(bullets, resolution.decisions)
        note = f"Re-running rewrite with your feedback: {resolution.feedback}"
        if locked:
            note += f" ({len(locked)} bullet(s) kept as-is.)"
        return Command(
            update=_narrate(
                {"tailoring_strategy": strategy, "rewritten_content": None, "locked_rewrites": locked},
                note,
            ),
            goto=JT_STEP5_REWRITE,
        )
    return Command(
        update=_narrate({}, "Stopped before document assembly. No documents were generated."),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Gate: step failed (retry or stop). Reached from steps 4 and 5 when their tool
# reported an error, so a failure is never shown as an empty-but-valid result.
# --------------------------------------------------------------------------- #

_RETRYABLE_STEPS = (
    JT_STEP2_COMPATIBILITY, JT_STEP4_SELECT, JT_STEP5_REWRITE, JT_STEP_TRANSLATE_CV, JT_STEP8_COVER_LETTER_CONTENT,
)


def route_after_select(state: OrchestratorState) -> str:
    return JT_GATE_STEP_FAILED if state.get("step_error") else JT_GATE_APPROVE_SELECTION


def route_after_rewrite(state: OrchestratorState) -> str:
    # Rewrites are fact-checked before the user sees them.
    return JT_GATE_STEP_FAILED if state.get("step_error") else JT_STEP5B_VERIFY


def jt_gate_step_failed(state: OrchestratorState) -> Command:
    error = state.get("step_error") or {}
    label = error.get("label") or "A step"
    payload = GatePayload(
        step="step_failed",
        kind="approval",
        narration=f"{label} failed. Approve to retry it, or reject to stop.",
        preview={"failed_step": label, "error": error.get("detail", "")},
        allowed_actions=["approve", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    failed_step = error.get("step")
    if resolution.action == "approve" and failed_step in _RETRYABLE_STEPS:
        return Command(
            update=_narrate({"step_error": None}, f"Retrying {label.lower()}."),
            goto=failed_step,
        )
    files_note = (
        "Your CV files are still available." if state.get("generated_files") else "No documents were generated."
    )
    return Command(
        update=_narrate({"step_error": None}, f"Stopped after {label.lower()} failed. {files_note}"),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Gate: confirm a change the chat proposed (re-run from a step, profile update)
#
# The API never changes the graph's state for these itself: it stores the
# proposal and routes here. Only an explicit approve applies it, and applying
# means restoring the state from before the target step and running forward
# from there - so every gate after that step is shown again. Reject returns to
# the gate the user was on.
# --------------------------------------------------------------------------- #

def jt_gate_confirm_change(state: OrchestratorState) -> Command:
    proposal = state.get("chat_proposal") or {}
    payload = GatePayload(
        step="confirm_change",
        kind="approval",
        narration=f"{proposal.get('summary') or 'Apply this change'}? Confirm to go ahead, or cancel to keep things as they are.",
        preview={
            "change": proposal.get("summary", ""),
            "what_happens": proposal.get("details", []),
        },
        allowed_actions=["approve", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    goto = proposal.get("goto")
    if resolution.action == "approve" and goto:
        update = {**(proposal.get("restore") or {}), "chat_proposal": None}
        return Command(update=_narrate(update, proposal.get("on_approve") or "Going ahead."), goto=goto)
    return Command(
        update=_narrate({"chat_proposal": None}, "Cancelled. Nothing was changed."),
        goto=proposal.get("return_to") or END,
    )


# --------------------------------------------------------------------------- #
# Step 6: assemble tailored CV (rewrites merged into a copy: tailored_cv_data)
# --------------------------------------------------------------------------- #

def jt_step6_assemble_cv(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
    rewritten = state.get("rewritten_content") or {}
    bullet_index = state.get("bullet_index") or _index_cv_bullets(cv_data)
    merged_cv, unplaced = _merge_rewrites_into_cv(
        cv_data, rewritten.get("rewritten_bullets", []) or [], bullet_index
    )

    if rewritten.get("updated_summary"):
        merged_cv["summary"] = rewritten["updated_summary"]
    left_out = state.get("left_out_bullet_ids") or []
    if left_out:
        merged_cv = _without_bullets(merged_cv, left_out, bullet_index)
    # The plan's headline went through the fact check and the user approved
    # it with the plan; it becomes the line under the name.
    headline = ((state.get("tailoring_strategy") or {}).get("plan") or {}).get("headline")
    if headline and not merged_cv.get("label"):
        merged_cv["label"] = headline

    narration = "Assembled the tailored CV. Generating PDF export."
    if unplaced:
        preview = "; ".join(t[:80] for t in unplaced[:3])
        narration += f"\n\nNote: {len(unplaced)} rewrite(s) could not be matched to a CV bullet and were skipped: {preview}"

    update: dict[str, Any] = {"tailored_cv_data": merged_cv}
    alignment = _alignment_after_tailoring(state, merged_cv)
    if alignment:
        update.update(alignment)
        narration += "\n\n" + _alignment_narration(alignment["alignment"])

    # cv_data stays the user's own CV: re-runs, the fact ledger and the
    # verifier compare against it.
    return _narrate(update, narration)


def _alignment_after_tailoring(state: OrchestratorState, tailored: dict[str, Any]) -> dict[str, Any] | None:
    """
    The rubric with its evidence searched again in the tailored CV, and the
    alignment before and after. Statuses don't change (the verifier stops a
    rewrite from adding a skill); what rewriting can do is show a skill the
    CV already has in a bullet, which the alignment rewards.
    """
    rubric = (state.get("compatibility_report") or {}).get("rubric") or []
    before = rubric_alignment(rubric)
    if not rubric or before is None:
        return None
    after_rows = refresh_evidence(rubric, build_fact_ledger(tailored, _index_cv_bullets(tailored)))
    newly_shown = [
        row.text for row, old in zip(after_rows, rubric)
        if row.category == "must_have" and row.evidence_bullet_ids and not old.get("evidence_bullet_ids")
    ]
    verifier = state.get("verifier_result") or {}
    return {
        "requirement_rubric_after": [row.model_dump() for row in after_rows],
        "alignment": {
            "before": before,
            "after": rubric_alignment(after_rows),
            "preservation": verifier.get("preservation_score"),
            "newly_shown": newly_shown,
        },
    }


def _alignment_narration(alignment: dict[str, Any]) -> str:
    before, after = alignment["before"], alignment["after"]
    line = f"Requirement alignment: {before:.2f} -> {after:.2f} ({after - before:+.2f})."
    if alignment.get("newly_shown"):
        line += " Now shown in a bullet: " + ", ".join(alignment["newly_shown"]) + "."
    if alignment.get("preservation") is not None:
        line += f" The rewrites kept {alignment['preservation']:.0%} of your original technologies and numbers."
    return line


# --------------------------------------------------------------------------- #
# CV language: asked right after scoring, then the whole CV is translated
# (and reviewed) before the plan, selection and rewriting, so every later step
# works on a CV in one language and the result can't come out mixed.
# --------------------------------------------------------------------------- #

_LANGUAGE_NAMES = {"english": "English", "german": "German"}
_POSTING_CODES = {"de": "german", "en": "english"}


def _posting_language(state: OrchestratorState) -> str:
    """The posting's language: as extracted, else read from its text, else
    the CV's own."""
    job = state.get("job_data") or {}
    code = str(job.get("posting_language") or "").strip().lower()[:2]
    if code in _POSTING_CODES:
        return _POSTING_CODES[code]
    text = " ".join(str(x) for key in ("responsibilities", "key_requirements", "qualifications")
                    for x in job.get(key) or [])
    detected = text_language(text)
    if detected:
        return {"de": "german", "en": "english"}[detected]
    return cv_content_language(state.get("cv_data") or {})


def _after_scoring(state: OrchestratorState) -> str:
    """Ask for the CV language once per session, then plan."""
    return JT_STEP3_STRATEGY if state.get("cv_language") else JT_GATE_CV_LANGUAGE


def jt_gate_cv_language(state: OrchestratorState) -> Command:
    posting = _posting_language(state)
    other = "english" if posting == "german" else "german"
    payload = GatePayload(
        step="cv_language",
        kind="choice",
        narration=(
            f"The posting is in {_LANGUAGE_NAMES[posting]}. Should your CV be in {_LANGUAGE_NAMES[posting]} "
            f"as well, or in {_LANGUAGE_NAMES[other]}? The whole CV will be in the language you pick."
        ),
        preview={"posting_language": _LANGUAGE_NAMES[posting]},
        allowed_actions=["choose"],
        choices=[posting, other],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    language = resolution.choice if resolution.choice in _LANGUAGE_NAMES else posting
    if needs_translation(state.get("cv_data") or {}, language):
        return Command(
            update=_narrate({"cv_language": language},
                            f"Your CV will be in {_LANGUAGE_NAMES[language]}. Translating it first."),
            goto=JT_STEP_TRANSLATE_CV,
        )
    return Command(
        update=_narrate({"cv_language": language},
                        f"Your CV will be in {_LANGUAGE_NAMES[language]}; it already is."),
        goto=JT_STEP3_STRATEGY,
    )


def route_after_translate(state: OrchestratorState) -> str:
    return JT_GATE_STEP_FAILED if state.get("step_error") else JT_GATE_APPROVE_TRANSLATION


def jt_step_translate_cv(state: OrchestratorState) -> dict[str, Any]:
    target = state.get("cv_language") or "english"
    cv_data = state.get("cv_data") or {}
    feedback = "\n".join(f"- {note}" for note in state.get("translation_feedback") or [] if note)
    raw = _safe_json_loads(translate_cv_fields.invoke({
        "fields_json": json.dumps(translatable_fields(cv_data), ensure_ascii=False),
        "language": target,
        "feedback": feedback,
    }), None)
    error = _tool_error(raw, "translation")
    if error:
        return _fail_step(JT_STEP_TRANSLATE_CV, "Translation", error)

    fields = {f.get("path"): f for f in raw.get("fields") or [] if isinstance(f, dict)}
    translated_cv, rows = translate_cv(cv_data, list(fields.values()))
    failed = [r for r in rows if r["problem"]]
    if failed:
        # One more try for just those fields, told what went wrong, so the
        # review doesn't show (and the CV doesn't keep) English leftovers that
        # a later step would translate anyway.
        notes = [f"- {r['path']}: {r['problem']}. Keep every technology name, number and year exactly."
                 for r in failed]
        retry = _safe_json_loads(translate_cv_fields.invoke({
            "fields_json": json.dumps([{"path": r["path"], "text": r["original"]} for r in failed], ensure_ascii=False),
            "language": target,
            "feedback": "\n".join(filter(None, [feedback, *notes])),
        }), None)
        if not _tool_error(retry, "translation"):
            fields.update({f.get("path"): f for f in retry.get("fields") or [] if isinstance(f, dict)})
            translated_cv, rows = translate_cv(cv_data, list(fields.values()))
    kept = [r for r in rows if r["problem"]]
    changed = [r for r in rows if not r["problem"] and r["translated"] != r["original"]]
    narration = f"Translated your CV into {_LANGUAGE_NAMES.get(target, target)}: {len(changed)} field(s) changed."
    if kept:
        narration += f" {len(kept)} kept their original text because the translation changed a fact; they're marked below."
    return _narrate(
        {"translation": {"target_language": target, "rows": rows, "translated_cv": translated_cv},
         "step_error": None},
        narration,
    )


def _translation_preview(translation: dict[str, Any]) -> dict[str, Any]:
    rows = translation.get("rows") or []
    return {
        "target_language": _LANGUAGE_NAMES.get(translation.get("target_language"), ""),
        "changes": [
            {"path": r["path"], "original": r["original"], "translated": r["translated"]}
            for r in rows if not r["problem"] and r["translated"] != r["original"]
        ],
        # Shown as "Projects 2 · description" with the proposed translation and
        # a plain reason, not as an internal path and a code.
        "kept_original": [
            {"path": r["path"], "original": r["original"], "translated": r["translated"], "problem": r["problem"]}
            for r in rows if r["problem"]
        ],
    }


def _with_cv(state: OrchestratorState, cv_data: dict[str, Any]) -> dict[str, Any]:
    """State update for a new version of the user's CV: its bullet index and
    fact ledger follow it (same ids), keeping what the user confirmed."""
    bullet_index = _index_cv_bullets(cv_data)
    confirmed = [f for f in (state.get("fact_ledger") or {}).values()
                 if isinstance(f, dict) and f.get("source") != "cv_extraction"]
    return {
        "cv_data": cv_data,
        "bullet_index": bullet_index,
        "fact_ledger": with_facts(build_fact_ledger(cv_data, bullet_index), confirmed),
    }


def jt_gate_approve_translation(state: OrchestratorState) -> Command:
    """
    approve: use the translated CV from here on; reject: keep the CV as it
    is; edit: translate again with the user's notes.
    """
    translation = state.get("translation") or {}
    language = _LANGUAGE_NAMES.get(translation.get("target_language"), "the chosen language")
    payload = GatePayload(
        step="approve_translation",
        kind="approval",
        narration=(
            f"Review the {language} version of your CV. Approve to tailor it in {language}, edit "
            "with notes to translate again, or reject to keep your CV as it is."
        ),
        preview=_translation_preview(translation),
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        translated = translation.get("translated_cv") or state.get("cv_data") or {}
        return Command(
            update=_narrate({**_with_cv(state, translated), "translation": None},
                            f"Using the {language} version of your CV."),
            goto=JT_STEP3_STRATEGY,
        )
    if resolution.action == "edit":
        notes = list(state.get("translation_feedback") or []) + [resolution.feedback or ""]
        return Command(
            update=_narrate({"translation_feedback": notes, "translation": None},
                            f"Translating again with your notes: {resolution.feedback}"),
            goto=JT_STEP_TRANSLATE_CV,
        )
    kept_language = cv_content_language(state.get("cv_data") or {})
    return Command(
        update=_narrate({"translation": None, "cv_language": kept_language},
                        "Keeping your CV as it is, without translating it."),
        goto=JT_STEP3_STRATEGY,
    )


def _tailored_cv(state: OrchestratorState) -> dict[str, Any]:
    """The CV to render: step 6's output. Sessions saved before step 6 kept
    its output separately still have the tailored CV in cv_data."""
    return state.get("tailored_cv_data") or state.get("cv_data") or {}


# --------------------------------------------------------------------------- #
# Step 7: generate CV PDF (LaTeX / Harvard template)
# --------------------------------------------------------------------------- #

def _with_skill_groups(cv_data: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """The CV with its skills as the plan's labelled groups (validated to hold
    only the CV's own skills). Without groups the renderer groups by skill
    family itself."""
    groups = [g for g in groups or [] if isinstance(g, dict) and g.get("skills")]
    if not groups:
        return cv_data
    other = "Weitere" if cv_content_language(cv_data) == "german" else "Other"
    return {**cv_data, "skills": [
        {"name": other if g.get("label") == "Other" else g.get("label", ""), "keywords": list(g["skills"])}
        for g in groups
    ]}


def _with_project_order(cv_data: dict[str, Any], order: list[int]) -> dict[str, Any]:
    """The CV with its projects in the plan's order; projects the plan didn't
    rank keep their place after the ranked ones."""
    projects = cv_data.get("projects")
    if not order or not isinstance(projects, list):
        return cv_data
    ranked = [projects[i] for i in order if isinstance(i, int) and 0 <= i < len(projects)]
    rest = [p for i, p in enumerate(projects) if i not in set(order)]
    return {**cv_data, "projects": ranked + rest}


def jt_step7_generate_cv_files(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    """Emit a tailored CV PDF from merged structured JSON."""
    cv_data = _tailored_cv(state)
    applicant_name = cv_data.get("name", "Applicant")
    base = _filename_base(cv_data)
    selected = state.get("selected_content") or {}
    plan = (state.get("tailoring_strategy") or {}).get("plan") or {}
    # Selection follows the plan's order and the user reviewed it there; the
    # plan's own order is the fallback.
    section_order = selected.get("section_order") or plan.get("section_order") or []

    generated: list[dict[str, str]] = list(state.get("generated_files") or [])
    messages: list[str] = []

    pdf_name = f"{base}_cv_tailored.pdf"
    pdf_msg = generate_cv_pdf.invoke({
        "cv_json": json.dumps(_with_skill_groups(
            _with_project_order(cv_data, plan.get("project_order") or []), plan.get("skill_groups") or []
        )),
        "output_filename": pdf_name,
        "applicant_name": applicant_name,
        "section_order_json": json.dumps(section_order),
        "output_dir": session_output_dir(config),
        "convention": plan.get("convention") or "international",
    })
    if isinstance(pdf_msg, str) and "Error" not in pdf_msg:
        filename = _extract_filename(pdf_msg) or pdf_name
        generated.append(file_record(filename, "cv", config))
        messages.append(f"PDF: {filename}")
        messages.extend(line for line in pdf_msg.splitlines() if line.startswith("Layout note:"))
    else:
        messages.append(f"PDF generation failed: {pdf_msg}")

    narration = "Generated CV files:\n" + "\n".join(messages) if messages else "No CV files were generated."
    return _narrate({"generated_files": generated}, narration)


# --------------------------------------------------------------------------- #
# Gate: cover letter language choice (en / de / skip)
# --------------------------------------------------------------------------- #

COVER_LETTER_CHOICES = ["english", "german", "skip"]


def jt_gate_cover_letter(state: OrchestratorState) -> Command:
    payload = GatePayload(
        step="cover_letter_language",
        kind="choice",
        narration=(
            "Would you like a matching cover letter? Pick a language or skip."
        ),
        preview={},
        allowed_actions=["choose"],
        choices=COVER_LETTER_CHOICES,
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    choice = (resolution.choice or "skip").lower()
    if choice not in COVER_LETTER_CHOICES:
        choice = "skip"

    if choice == "skip":
        return Command(
            update=_narrate({"cover_letter_language": None}, "All done. Skipping the cover letter."),
            goto=END,
        )
    return Command(
        update={
            "pending_gate": None,
            "cover_letter_language": choice,  # type: ignore[typeddict-item]
        },
        goto=JT_STEP_SET_COVER_LETTER_RECIPIENT,
    )


# --------------------------------------------------------------------------- #
# Resolve cover letter recipient (from job posting or user input)
# --------------------------------------------------------------------------- #

def jt_step_set_cover_letter_recipient(state: OrchestratorState) -> Command:
    recipient = extract_recipient_name(state.get("job_data"))
    if recipient:
        return Command(
            update=_narrate(
                {"cover_letter_recipient": recipient},
                f"Addressing the cover letter to {recipient}.",
            ),
            goto=JT_STEP8_COVER_LETTER_CONTENT,
        )
    return Command(
        update={"pending_gate": None},
        goto=JT_GATE_COVER_LETTER_RECIPIENT,
    )


def jt_gate_cover_letter_recipient(state: OrchestratorState) -> Command:
    payload = GatePayload(
        step="cover_letter_recipient",
        kind="input",
        narration=(
            "Who should the cover letter be addressed to? "
            "Enter the recipient's name (no default is used)."
        ),
        preview={},
        allowed_actions=["edit"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)
    recipient = (resolution.feedback or "").strip()
    if not recipient:
        return Command(
            update=_narrate({}, "Please provide the recipient name to continue."),
            goto=JT_GATE_COVER_LETTER_RECIPIENT,
        )
    return Command(
        update={
            "pending_gate": None,
            "cover_letter_recipient": recipient,  # type: ignore[typeddict-item]
        },
        goto=JT_STEP8_COVER_LETTER_CONTENT,
    )


# --------------------------------------------------------------------------- #
# Step 8: cover letter content
# --------------------------------------------------------------------------- #

def jt_step8_cover_letter_content(state: OrchestratorState) -> dict[str, Any]:
    cv_json = json.dumps(_tailored_cv(state))
    job_json = json.dumps(state.get("job_data") or {})
    company_json = json.dumps(state.get("company_data") or {}) if state.get("company_data") else ""
    language = state.get("cover_letter_language") or "english"

    content_json = generate_cover_letter_content.invoke({
        "cv_json": cv_json,
        "job_json": job_json,
        "company_json": company_json,
        "language": language,
    })
    content = _safe_json_loads(content_json, None)
    error = _tool_error(content, "cover letter")
    if error:
        return _fail_step(JT_STEP8_COVER_LETTER_CONTENT, "Cover letter drafting", error)

    narration_lines = [
        f"Cover letter drafted ({language}). Review the paragraphs and approve to export."
    ]
    if content.get("betreff"):
        narration_lines.append(f"Betreff: {content['betreff']}")

    return _narrate({"cover_letter_content": content, "step_error": None}, "\n".join(narration_lines))


def route_after_cover_letter(state: OrchestratorState) -> str:
    return JT_GATE_STEP_FAILED if state.get("step_error") else JT_GATE_APPROVE_COVER_LETTER


# --------------------------------------------------------------------------- #
# Gate: approve cover letter
# --------------------------------------------------------------------------- #

def jt_gate_approve_cover_letter(state: OrchestratorState) -> Command:
    payload = GatePayload(
        step="approve_cover_letter",
        kind="approval",
        narration="Approve to export the cover letter, or edit to regenerate.",
        preview=state.get("cover_letter_content") or {},
        allowed_actions=["approve", "edit", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        return Command(
            update={"pending_gate": None},
            goto=JT_STEP9_COVER_LETTER_FILES,
        )
    if resolution.action == "edit":
        company = state.get("company_data") or {}
        company["user_feedback"] = (company.get("user_feedback") or []) + [resolution.feedback or ""]
        return Command(
            update=_narrate(
                {"company_data": company, "cover_letter_content": None},
                f"Regenerating the cover letter with your feedback: {resolution.feedback}",
            ),
            goto=JT_STEP8_COVER_LETTER_CONTENT,
        )
    return Command(
        update=_narrate({}, "Stopped before cover-letter export. CV files (if generated) are still available."),
        goto=END,
    )


# --------------------------------------------------------------------------- #
# Step 9: cover letter PDF (LaTeX)
# --------------------------------------------------------------------------- #

def jt_step9_cover_letter_files(state: OrchestratorState, config: RunnableConfig) -> dict[str, Any]:
    cv_data = _tailored_cv(state)
    content = state.get("cover_letter_content") or {}
    language = state.get("cover_letter_language") or "english"
    base = _filename_base(cv_data)
    applicant_name = cv_data.get("name", "Applicant")
    applicant_contact = _applicant_contact(cv_data)
    recipient_info = (state.get("cover_letter_recipient") or "").strip()
    if not recipient_info:
        return _narrate(
            {},
            "Cover-letter export skipped: recipient name is required.",
        )

    suffix = "anschreiben" if language == "german" else "cover_letter"
    generated: list[dict[str, str]] = list(state.get("generated_files") or [])
    messages: list[str] = []

    pdf_name = f"{base}_{suffix}.pdf"
    pdf_msg = generate_cover_letter_pdf.invoke({
        "content_json": json.dumps(content),
        "output_filename": pdf_name,
        "applicant_name": applicant_name,
        "applicant_contact": applicant_contact,
        "recipient_info": recipient_info,
        "cv_json": json.dumps(cv_data),
        "job_json": json.dumps(state.get("job_data") or {}),
        "output_dir": session_output_dir(config),
    })
    if isinstance(pdf_msg, str) and "Error" not in pdf_msg:
        filename = _extract_filename(pdf_msg) or pdf_name
        generated.append(file_record(filename, "cover_letter", config))
        messages.append(f"PDF: {filename}")
    else:
        messages.append(f"Cover-letter PDF failed: {pdf_msg}")

    narration = (
        "Cover letter exported:\n" + "\n".join(messages)
        if messages else "No cover-letter files were generated."
    )
    return _narrate({"generated_files": generated}, narration)
