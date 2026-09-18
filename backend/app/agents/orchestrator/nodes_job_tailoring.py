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
from app.services.latex_renderer import extract_recipient_name
from app.agents.scoring_agent import calculate_compatibility_score_v2


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
# Every bullet-like string in the CV gets a stable id of the form
# "<section>.<entry_idx>.<field>.<idx>" (or "summary" for the top-level
# summary). The id survives selection -> rewrite -> merge so a rewrite is
# matched to its source bullet even if the model's echoed `original` text
# drifts slightly, and so the UI can address individual bullets for
# accept/keep-original/custom decisions.
#
# Field coverage mirrors what `resume_adapter.build_harvard_sections` (the
# LaTeX renderer's input) actually reads, so every id here is guaranteed to
# land somewhere in the generated PDF.
# --------------------------------------------------------------------------- #

_BULLET_FIELDS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "experience": ("responsibilities",),
    "projects": ("outcomes",),
    "leadership_activities": ("highlights",),
}
_TEXT_FIELDS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "projects": ("description",),
    "leadership_activities": ("description",),
}


def _normalize_bullet_text(text: str) -> str:
    """Casefold, collapse whitespace, strip trailing punctuation - used as a
    fallback match key when an id round-trip fails or the model paraphrases
    the echoed original slightly."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed.rstrip(".,;: ").casefold()


def _index_cv_bullets(cv_data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """
    Build {bullet_id: {section, entry_idx, field, idx, text}} for every
    bullet/description string in the CV, plus the top-level summary.
    """
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(cv_data, dict):
        return index

    for section, fields in _BULLET_FIELDS_BY_SECTION.items():
        entries = cv_data.get(section)
        if not isinstance(entries, list):
            continue
        for entry_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            for field in fields:
                values = entry.get(field)
                if not isinstance(values, list):
                    continue
                for idx, text in enumerate(values):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    bullet_id = f"{section}.{entry_idx}.{field}.{idx}"
                    index[bullet_id] = {
                        "section": section,
                        "entry_idx": entry_idx,
                        "field": field,
                        "idx": idx,
                        "text": text,
                    }
            for field in _TEXT_FIELDS_BY_SECTION.get(section, ()):
                text = entry.get(field)
                if isinstance(text, str) and text.strip():
                    bullet_id = f"{section}.{entry_idx}.{field}"
                    index[bullet_id] = {
                        "section": section,
                        "entry_idx": entry_idx,
                        "field": field,
                        "idx": None,
                        "text": text,
                    }

    summary = cv_data.get("summary")
    if isinstance(summary, str) and summary.strip():
        index["summary"] = {"section": "summary", "entry_idx": None, "field": "summary", "idx": None, "text": summary}

    return index


def _bullet_index_prompt_block(bullet_index: dict[str, dict[str, Any]]) -> str:
    """Render the id index as a numbered `[id] text` list for LLM prompts."""
    lines = [f"[{bullet_id}] {ref['text']}" for bullet_id, ref in bullet_index.items()]
    return "\n".join(lines) if lines else "(no bullets found)"


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


def _file_record(filename: str, file_type: str) -> dict[str, str]:
    return {
        "filename": filename,
        "file_type": file_type,
        "download_url": f"/api/files/{filename}",
    }


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
    cv_data = state.get("cv_data") or {}
    job_data = state.get("job_data") or {}
    name = cv_data.get("name", "you")
    job_title = job_data.get("job_title") or "the target role"

    narration = (
        f"Starting the job-tailoring flow for {name} against {job_title}. "
        "I'll summarise the posting, score the match, propose a tailoring "
        "strategy, then walk you through bullet selection and rewriting "
        "with explicit approvals before any documents are generated."
    )
    return _narrate({"bullet_index": _index_cv_bullets(cv_data)}, narration)


# --------------------------------------------------------------------------- #
# Step 1: summarise job posting
# --------------------------------------------------------------------------- #

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
    report = _safe_json_loads(report_json, {
        "aggregate_score": 0.0,
        "level": "unknown",
        "interpretation": "Compatibility scoring failed.",
        "dimensions": [],
        "gap_analysis": {"matched_skills": [], "transferable_skills": [], "missing_skills": [], "over_qualified_signals": []},
        "warnings": ["scoring v2 returned non-JSON output"],
    })

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
    matched = gap.get("matched_skills", []) or []
    transferable = gap.get("transferable_skills", []) or []
    missing = gap.get("missing_skills", []) or []
    lines.append("")
    lines.append(f"Matched: {len(matched)} | Transferable: {len(transferable)} | Missing: {len(missing)}")
    # Name the actual gaps instead of only a count - a bare "Missing: 3" tells
    # the user nothing they can act on.
    missing_names = [m.get("required_skill") for m in missing if isinstance(m, dict) and m.get("required_skill")]
    if missing_names:
        lines.append(f"Missing skills: {', '.join(missing_names)}")
    transferable_bridges = [
        f"{m.get('required_skill')} (via {m.get('matched_with')})"
        for m in transferable
        if isinstance(m, dict) and m.get("required_skill") and m.get("bridge_bullet")
    ]
    if transferable_bridges:
        lines.append(f"Bridgeable via existing experience: {', '.join(transferable_bridges)}")
    if report.get("interpretation"):
        lines.append("")
        lines.append(report["interpretation"])
    if missing_names and not transferable_bridges:
        lines.append(
            "\nNote: none of these gaps can be bridged from your current CV content - "
            "the rewrite step can only rephrase what's already there, not invent missing "
            "skills. If you have any real exposure to "
            f"{', '.join(missing_names)} (a course, a side project, brief use on the job), "
            "add it to your CV before this step so it can be surfaced."
        )

    return _narrate({"compatibility_report": report}, "\n".join(lines))


# --------------------------------------------------------------------------- #
# Gate: present score (soft gate - user just acknowledges before strategy)
# --------------------------------------------------------------------------- #

def jt_gate_present_score(state: OrchestratorState) -> Command:
    report = state.get("compatibility_report") or {}
    payload = GatePayload(
        step="present_score",
        kind="approval",
        narration="Proceed to design the tailoring strategy?",
        preview={
            "aggregate_score": report.get("aggregate_score"),
            "level": report.get("level"),
            "dimensions": report.get("dimensions", []),
            "gap_analysis": report.get("gap_analysis", {}),
            "interpretation": report.get("interpretation", ""),
        },
        allowed_actions=["approve", "reject"],
    ).model_dump()

    raw = interrupt(payload)
    resolution = GateResolution.model_validate(raw)

    if resolution.action == "approve":
        return Command(
            update={"pending_gate": None},
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

    focus = ", ".join(strategy.get("focus_areas", []) or []) or "none"
    lines = [
        "Tailoring strategy",
        "",
        f"Strategy: {strategy.get('strategy', '')}",
        f"Intensity: {strategy.get('intensity', 'moderate')}",
        f"Focus areas: {focus}",
        f"Directives: {len(strategy.get('directives', []) or [])}",
    ]
    return _narrate({"tailoring_strategy": strategy}, "\n".join(lines))


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
    })
    selected = _safe_json_loads(selected_json, {
        "selected_bullets": [],
        "section_order": [],
        "sections_to_emphasize": [],
        "items_to_de_emphasize": [],
    })

    # Drop bullets the model matched to an id that doesn't exist - never
    # trust an echoed id blindly (guards against hallucinated ids).
    bullets = selected.get("selected_bullets", []) or []
    for b in bullets:
        if isinstance(b, dict) and isinstance(b.get("bullet_id"), str) and b["bullet_id"] not in bullet_index:
            b["bullet_id"] = None

    fallback_used = False
    if not bullets and bullet_index:
        # The model returned nothing to work with, despite the CV having
        # content - a low-relevance CV/job pairing is not a reason to skip
        # tailoring entirely. Fall back to the CV's own bullets so rewrite
        # still has something real to work on, flagged with an honest
        # low relevance score rather than a fabricated one.
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
    selected["selected_bullets"] = bullets

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
            "\nNote: nothing in your CV closely matched this job, so I fell back to your CV's "
            "own strongest bullets rather than returning nothing."
        )
    return _narrate({"selected_content": selected, "bullet_index": bullet_index}, "\n".join(lines))


# --------------------------------------------------------------------------- #
# Gate: approve selection (HARD GATE - edit loops back to step 4)
# --------------------------------------------------------------------------- #

def _finalize_selection_decisions(
    selected_bullets: list[dict[str, Any]],
    decisions: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply per-bullet include/exclude decisions. Default: include (so a
    plain Approve with no decisions keeps today's behaviour)."""
    decisions = decisions or {}
    kept: list[dict[str, Any]] = []
    for b in selected_bullets:
        if not isinstance(b, dict):
            continue
        bullet_id = b.get("bullet_id")
        decision = decisions.get(bullet_id) if bullet_id else None
        choice = getattr(decision, "choice", None) if decision else "include"
        if choice == "exclude":
            continue
        kept.append(b)
    return kept


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
        finalized = _finalize_selection_decisions(selected_bullets, resolution.decisions)
        excluded_count = len(selected_bullets) - len(finalized)
        updated_selected = dict(selected)
        updated_selected["selected_bullets"] = finalized
        update: dict[str, Any] = {"pending_gate": None, "selected_content": updated_selected}
        if excluded_count:
            update = _narrate(
                update,
                f"Continuing with {len(finalized)} bullets ({excluded_count} excluded).",
            )
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
    })
    rewritten = _safe_json_loads(rewritten_json, {
        "rewritten_bullets": [],
        "updated_summary": "",
        "keywords_inserted": [],
    })

    bullets = rewritten.get("rewritten_bullets", []) or []
    selected_bullets = (state.get("selected_content") or {}).get("selected_bullets", []) or []
    fallback_used = False
    if not bullets and selected_bullets:
        # The rewrite model returned nothing despite being given bullets to
        # work with. Echo the selections back unchanged rather than silently
        # producing a no-op tailoring pass - the user can still see and
        # accept/edit them individually at the gate.
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
    lines = [
        "Rewritten content",
        "",
        f"Bullets rewritten: {len(bullets)}",
        f"Average rewrite confidence: {avg_conf:.2f}",
        f"Keywords woven in: {', '.join(rewritten.get('keywords_inserted', []) or []) or '(none)'}",
    ]
    if fallback_used:
        lines.append(
            "\nNote: the rewrite step returned nothing usable, so the selected bullets are shown "
            "unchanged below - review and edit them at the gate if you'd like different wording."
        )
    return _narrate({"rewritten_content": rewritten, "locked_rewrites": None}, "\n".join(lines))


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
        bullet_id = b.get("bullet_id")
        decision = decisions.get(bullet_id) if bullet_id else None
        if decision is None:
            continue
        choice = getattr(decision, "choice", None)
        if choice == "keep_original":
            locked.append({**b, "rewritten": b.get("original", ""), "keywords_added": []})
        elif choice == "custom" and getattr(decision, "text", None):
            locked.append({**b, "rewritten": decision.text})
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
# Step 6: assemble tailored CV (merge rewrites into cv_data)
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

    narration = "Assembled the tailored CV. Generating PDF export."
    if unplaced:
        preview = "; ".join(t[:80] for t in unplaced[:3])
        narration += f"\n\nNote: {len(unplaced)} rewrite(s) could not be matched to a CV bullet and were skipped: {preview}"

    return _narrate(
        {"cv_data": merged_cv},
        narration,
    )


# --------------------------------------------------------------------------- #
# Step 7: generate CV PDF (LaTeX / Harvard template)
# --------------------------------------------------------------------------- #

def jt_step7_generate_cv_files(state: OrchestratorState) -> dict[str, Any]:
    """Emit a tailored CV PDF from merged structured JSON."""
    cv_data = state.get("cv_data") or {}
    applicant_name = cv_data.get("name", "Applicant")
    base = _filename_base(cv_data)
    selected = state.get("selected_content") or {}
    section_order = selected.get("section_order") or []

    generated: list[dict[str, str]] = list(state.get("generated_files") or [])
    messages: list[str] = []

    pdf_name = f"{base}_cv_tailored.pdf"
    pdf_msg = generate_cv_pdf.invoke({
        "cv_json": json.dumps(cv_data),
        "output_filename": pdf_name,
        "applicant_name": applicant_name,
        "section_order_json": json.dumps(section_order),
    })
    if isinstance(pdf_msg, str) and "Error" not in pdf_msg:
        filename = _extract_filename(pdf_msg) or pdf_name
        generated.append(_file_record(filename, "cv"))
        messages.append(f"PDF: {filename}")
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
    cv_json = json.dumps(state.get("cv_data") or {})
    job_json = json.dumps(state.get("job_data") or {})
    company_json = json.dumps(state.get("company_data") or {}) if state.get("company_data") else ""
    language = state.get("cover_letter_language") or "english"

    content_json = generate_cover_letter_content.invoke({
        "cv_json": cv_json,
        "job_json": job_json,
        "company_json": company_json,
        "language": language,
    })
    content = _safe_json_loads(content_json, {})

    narration_lines = [
        f"Cover letter drafted ({language}). Review the paragraphs and approve to export."
    ]
    if content.get("betreff"):
        narration_lines.append(f"Betreff: {content['betreff']}")

    return _narrate({"cover_letter_content": content}, "\n".join(narration_lines))


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

def jt_step9_cover_letter_files(state: OrchestratorState) -> dict[str, Any]:
    cv_data = state.get("cv_data") or {}
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
    })
    if isinstance(pdf_msg, str) and "Error" not in pdf_msg:
        filename = _extract_filename(pdf_msg) or pdf_name
        generated.append(_file_record(filename, "cover_letter"))
        messages.append(f"PDF: {filename}")
    else:
        messages.append(f"Cover-letter PDF failed: {pdf_msg}")

    narration = (
        "Cover letter exported:\n" + "\n".join(messages)
        if messages else "No cover-letter files were generated."
    )
    return _narrate({"generated_files": generated}, narration)
