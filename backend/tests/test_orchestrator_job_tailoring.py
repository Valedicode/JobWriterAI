"""
End-to-end harness for the job-tailoring branch of the orchestrator graph.

Every LLM-backed @tool is replaced with a deterministic stub so the test runs
offline. We drive the graph through each gate, verify the interrupt payloads
have the expected ``step`` identifiers, and confirm the final state contains
both the tailored CV files and the cover-letter files.

Run with::

    cd backend
    pytest tests/test_orchestrator_job_tailoring.py -v
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.orchestrator.graph import build_graph
from app.agents.orchestrator.nodes_job_tailoring import (
    JT_ENTRY,
    JT_GATE_APPROVE_COVER_LETTER,
    JT_GATE_APPROVE_REWRITE,
    JT_GATE_APPROVE_SELECTION,
    JT_GATE_COVER_LETTER,
    JT_GATE_PRESENT_SCORE,
    route_after_score,
)


# --------------------------------------------------------------------------- #
# Fixed deterministic fixture data
# --------------------------------------------------------------------------- #

CV = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": "+1 555 0100",
    "skills": ["Python", "Mathematics"],
    "education": ["B.Sc. Computer Science"],
    "experience": [
        {
            "position": "Engineer",
            "company": "Analytical Engines Ltd.",
            "responsibilities": [
                "Designed the first algorithm",
                "Wrote technical notes",
            ],
        }
    ],
    "projects": [],
    "leadership_activities": [],
}

JOB = {
    "job_title": "Senior Python Engineer",
    "job_level": "Senior",
    "required_skills": ["Python", "Algorithms"],
    "preferred_skills": ["Mathematics"],
    "employment_type": "Full-time",
    "location": "Remote",
    "responsibilities": ["Build new things"],
    "qualifications": [],
    "key_requirements": ["5+ yoe"],
    "recipient_name": "Dr. Jane Smith",
}

JOB_SUMMARY = {
    "role": "Senior Python Engineer",
    "responsibilities": ["Build new things"],
    "required_skills": ["Python", "Algorithms"],
    "preferred_skills": ["Mathematics"],
    "key_notes": ["Remote"],
}

COMPATIBILITY_REPORT = {
    "aggregate_score": 0.78,
    "level": "high",
    "interpretation": "Strong match.",
    "dimensions": [
        {"name": "hard_skills", "score": 0.85, "weight": 0.40, "rationale": ""},
        {"name": "experience", "score": 0.70, "weight": 0.15, "rationale": ""},
        {"name": "seniority", "score": 0.75, "weight": 0.15, "rationale": ""},
        {"name": "domain", "score": 0.65, "weight": 0.15, "rationale": ""},
        {"name": "ats_keywords", "score": 0.80, "weight": 0.15, "rationale": ""},
    ],
    "gap_analysis": {
        "matched_skills": [
            {"required_skill": "Python", "matched_with": "Python", "kind": "direct",
             "transferability": 1.0, "rationale": "", "bridge_bullet": None, "is_required": True}
        ],
        "transferable_skills": [],
        "missing_skills": [],
        "over_qualified_signals": [],
    },
    "warnings": [],
}

STRATEGY = {
    "scoring_version": "v2",
    "aggregate_score": 0.78,
    "level": "high",
    "intensity": "minor",
    "strategy": "Polish keywords and ordering.",
    "focus_areas": ["inject_ats_keywords"],
    "directives": [
        {"type": "inject_keywords", "section": "skills_and_summary", "guidance": "Add Python explicitly."}
    ],
    "summary": {"matched_skills_count": 1, "transferable_skills_count": 0,
                "missing_skills_count": 0, "over_qualified_signals_count": 0},
}

SELECTED = {
    "selected_bullets": [
        {"section": "experience", "original_text": "Designed the first algorithm",
         "relevance_score": 0.92, "reason": "matches algorithms keyword"},
    ],
    "section_order": ["experience", "skills", "education"],
    "sections_to_emphasize": ["experience"],
    "items_to_de_emphasize": [],
}

REWRITTEN = {
    "rewritten_bullets": [
        {
            "original": "Designed the first algorithm",
            "rewritten": "Designed and validated the first published algorithm in Python idioms.",
            "confidence": 0.88,
            "keywords_added": ["Python"],
        }
    ],
    "updated_summary": "",
    "keywords_inserted": ["Python", "Algorithms"],
}

COVER_LETTER_CONTENT = {
    "language": "english",
    "betreff": "",
    "grussformel": "",
    "opening_paragraph": "Hello.",
    "body_paragraph_1": "I match well.",
    "body_paragraph_2": "I match more.",
    "body_paragraph_3": "",
    "closing_paragraph": "Thank you.",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Pull the first interrupt's value out of an invoke() result."""
    interrupts = result.get("__interrupt__")
    assert interrupts, f"expected an interrupt; got {result}"
    first = interrupts[0]
    return first.value if hasattr(first, "value") else first["value"]


def _fake_tool(fn):
    """Wrap a function as a SimpleNamespace with an .invoke method.

    LangChain ``StructuredTool`` instances are frozen Pydantic models, so we
    cannot patch ``.invoke`` on them directly. Replacing the entire bound
    reference inside the node module with a SimpleNamespace gives nodes the
    same ``.invoke(dict)`` interface without touching the real tools.
    """
    return SimpleNamespace(invoke=fn)


@pytest.fixture
def patched_tools():
    """Deterministic stand-ins for every LLM/PDF-backed tool referenced by
    the job-tailoring nodes."""
    stubs = {
        "generate_job_summary": _fake_tool(lambda _: json.dumps(JOB_SUMMARY)),
        "calculate_compatibility_score_v2": _fake_tool(lambda _: json.dumps(COMPATIBILITY_REPORT)),
        "decide_tailoring_strategy": _fake_tool(lambda _: json.dumps(STRATEGY)),
        "select_prioritize_content": _fake_tool(lambda _: json.dumps(SELECTED)),
        "rewrite_enhance_content": _fake_tool(lambda _: json.dumps(REWRITTEN)),
        "generate_cv_pdf": _fake_tool(
            lambda kw: f"CV PDF generated successfully! The file '{kw['output_filename']}' is ready for download."
        ),
        "generate_cover_letter_content": _fake_tool(lambda _: json.dumps(COVER_LETTER_CONTENT)),
        "generate_cover_letter_pdf": _fake_tool(
            lambda kw: f"Cover letter PDF generated successfully! The file '{kw['output_filename']}' is ready for download."
        ),
    }
    contexts = [
        patch(f"app.agents.orchestrator.nodes_job_tailoring.{name}", stub)
        for name, stub in stubs.items()
    ]
    for c in contexts:
        c.start()
    yield
    for c in contexts:
        c.stop()


@pytest.fixture
def graph():
    """Fresh in-memory graph per test."""
    return build_graph(checkpointer=InMemorySaver())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_full_happy_path_emits_all_five_gates_and_generates_files(graph, patched_tools):
    config = {"configurable": {"thread_id": "test-happy"}}

    # 1. Initial invocation runs entry + steps 1-2, then interrupts at present_score.
    result = graph.invoke(
        {"flow": "job_tailoring", "cv_data": CV, "job_data": JOB},
        config=config,
    )
    payload = _interrupt_payload(result)
    assert payload["step"] == "present_score"
    assert payload["kind"] == "approval"
    assert "aggregate_score" in payload["preview"]

    # 2. Approve compatibility -> runs step 3 + 4, interrupts at approve_selection.
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_selection"
    assert payload["preview"]["selected_bullets"], "selected content should be in preview"

    # 3. Approve selection -> runs step 5, interrupts at approve_rewrite.
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    assert payload["preview"]["rewritten_bullets"][0]["rewritten"].startswith("Designed and validated")

    # 4. Approve rewrite -> runs step 6 + 7, interrupts at cover_letter language.
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "cover_letter_language"
    assert payload["kind"] == "choice"
    assert "english" in payload["choices"]

    # 5. Pick 'english' -> runs step 8 (cover-letter content), interrupts at approve_cover_letter.
    result = graph.invoke(Command(resume={"action": "choose", "choice": "english"}), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_cover_letter"
    assert payload["preview"]["opening_paragraph"] == "Hello."

    # 6. Approve cover letter -> step 9, then END.
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert "__interrupt__" not in result, "graph should terminate after step 9"

    state = graph.get_state(config).values
    files = state["generated_files"]
    types = sorted(f["file_type"] for f in files)
    names = [f["filename"] for f in files]
    assert "cv" in types and "cover_letter" in types
    assert any(n.endswith("_cv_tailored.pdf") for n in names)
    assert any(n.endswith("_cover_letter.pdf") for n in names)

    # The first rewritten bullet should have been merged into experience.
    merged_bullet = state["tailored_cv_data"]["experience"][0]["responsibilities"][0]
    assert merged_bullet.startswith("Designed and validated")


def test_edit_at_approve_selection_reruns_step_4(graph, patched_tools):
    """An 'edit' action on approve_selection should loop back to step 4."""
    config = {"configurable": {"thread_id": "test-edit-selection"}}

    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    graph.invoke(Command(resume={"action": "approve"}), config=config)  # past present_score

    # Now we're paused at approve_selection. Edit it.
    result = graph.invoke(
        Command(resume={"action": "edit", "feedback": "Use more action verbs."}),
        config=config,
    )
    payload = _interrupt_payload(result)
    # After edit, step 4 reruns and we land on approve_selection again.
    assert payload["step"] == "approve_selection"

    # The feedback should have been recorded in the strategy.
    state = graph.get_state(config).values
    feedback = state["tailoring_strategy"]["user_feedback"]
    assert feedback[-1]["stage"] == "approve_selection"
    assert "action verbs" in feedback[-1]["text"]


def test_reject_at_present_score_terminates_without_files(graph, patched_tools):
    """A 'reject' on the first soft gate should end the graph immediately."""
    config = {"configurable": {"thread_id": "test-reject-score"}}

    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    result = graph.invoke(Command(resume={"action": "reject"}), config=config)

    assert "__interrupt__" not in result
    state = graph.get_state(config).values
    assert state.get("generated_files") in (None, [])
    assert "Stopped" in state["last_narration"]


def test_missing_recipient_prompts_input_gate(graph, patched_tools):
    """When the job posting has no named contact, ask the user for the recipient."""
    config = {"configurable": {"thread_id": "test-recipient-gate"}}
    job_without_recipient = {k: v for k, v in JOB.items() if k != "recipient_name"}

    graph.invoke(
        {"flow": "job_tailoring", "cv_data": CV, "job_data": job_without_recipient},
        config=config,
    )
    graph.invoke(Command(resume={"action": "approve"}), config=config)
    graph.invoke(Command(resume={"action": "approve"}), config=config)
    graph.invoke(Command(resume={"action": "approve"}), config=config)
    result = graph.invoke(Command(resume={"action": "choose", "choice": "english"}), config=config)

    payload = _interrupt_payload(result)
    assert payload["step"] == "cover_letter_recipient"
    assert payload["kind"] == "input"

    result = graph.invoke(
        Command(resume={"action": "edit", "feedback": "Dr. Jane Smith"}),
        config=config,
    )
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_cover_letter"

    state = graph.get_state(config).values
    assert state["cover_letter_recipient"] == "Dr. Jane Smith"


def test_skip_cover_letter_after_cv_generation(graph, patched_tools):
    """Picking 'skip' at the cover-letter gate should end after CV files only."""
    config = {"configurable": {"thread_id": "test-skip-cover"}}

    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
    graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection
    graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_rewrite
    result = graph.invoke(Command(resume={"action": "choose", "choice": "skip"}), config=config)

    assert "__interrupt__" not in result
    state = graph.get_state(config).values
    files = state["generated_files"]
    types = {f["file_type"] for f in files}
    assert "cover_letter" not in types
    # Only the CV PDF was requested.
    assert any(f["filename"].endswith("_cv_tailored.pdf") for f in files)


def test_discovery_flow_routes_to_stub(graph, patched_tools):
    config = {"configurable": {"thread_id": "test-discovery"}}
    result = graph.invoke({"flow": "discovery", "cv_data": CV}, config=config)
    assert "not yet available" in result["last_narration"]


# NOTE: the previous test asserted cv_review routed to a stub. The cv_review
# branch is now a full 6-step graph; coverage moved to
# tests/test_orchestrator_cv_review.py.


# --------------------------------------------------------------------------- #
# Phase 1: bullet identity - robust merge, unplaced warnings, summary merge
# --------------------------------------------------------------------------- #

def test_rewrite_merges_by_bullet_id_even_when_echoed_text_drifts(graph, patched_tools):
    """A rewrite whose original text drifts from the CV (extra period) must
    still land in the PDF-bound cv_data, because bullet_id is authoritative."""
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-drift-merge"}}
    selected_with_id = {
        "selected_bullets": [
            {"section": "experience", "original_text": "Designed the first algorithm",
             "relevance_score": 0.92, "reason": "matches", "bullet_id": "experience.0.responsibilities.0"},
        ],
        "section_order": ["experience"],
        "sections_to_emphasize": [],
        "items_to_de_emphasize": [],
    }
    rewritten_with_id = {
        "rewritten_bullets": [
            {
                "original": "Designed the first algorithm.",  # drifted: trailing period added
                "rewritten": "Designed and validated the first published algorithm.",
                "confidence": 0.9,
                "keywords_added": [],
                "bullet_id": "experience.0.responsibilities.0",
            }
        ],
        "updated_summary": "",
        "keywords_inserted": [],
    }

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fake_tool(lambda _: json.dumps(selected_with_id)),
    ), _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(rewritten_with_id)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_rewrite

    state = graph.get_state(config).values
    merged_bullet = state["tailored_cv_data"]["experience"][0]["responsibilities"][0]
    assert merged_bullet.startswith("Designed and validated the first published algorithm")


def test_rewrite_with_unmatchable_bullet_is_dropped_and_narrated(graph, patched_tools):
    """A rewrite that matches nothing in the CV must not silently vanish -
    it is skipped and reported in the assembly narration."""
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-unplaced"}}
    rewritten_bogus = {
        "rewritten_bullets": [
            {
                "original": "This text does not exist anywhere in the CV",
                "rewritten": "Fabricated rewrite",
                "confidence": 0.5,
                "keywords_added": [],
                "bullet_id": "experience.99.responsibilities.0",
            }
        ],
        "updated_summary": "",
        "keywords_inserted": [],
    }

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(rewritten_bogus)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_rewrite

    state = graph.get_state(config).values
    # Original CV text is untouched.
    assert state["tailored_cv_data"]["experience"][0]["responsibilities"][0] == "Designed the first algorithm"
    # step6's narration (about the unplaced rewrite) is superseded by step7's
    # as `last_narration`, but it must still be in the message transcript.
    message_texts = [getattr(m, "content", "") for m in state["messages"]]
    assert any("could not be matched" in t for t in message_texts)


def test_updated_summary_is_applied_to_cv_data(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-updated-summary"}}
    rewritten_with_summary = dict(REWRITTEN)
    rewritten_with_summary["updated_summary"] = "Python engineer with a strong algorithms background."

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(rewritten_with_summary)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)

    state = graph.get_state(config).values
    assert state["tailored_cv_data"]["summary"] == "Python engineer with a strong algorithms background."


# --------------------------------------------------------------------------- #
# Phase 4: per-bullet decisions on the selection and rewrite gates
# --------------------------------------------------------------------------- #

TWO_BULLET_CV = {
    **CV,
    "experience": [
        {
            "position": "Engineer",
            "company": "Analytical Engines Ltd.",
            "responsibilities": [
                "Designed the first algorithm",
                "Wrote technical notes",
            ],
        }
    ],
}

TWO_BULLET_REWRITTEN = {
    "rewritten_bullets": [
        {
            "original": "Designed the first algorithm",
            "rewritten": "Designed and validated the first published algorithm.",
            "confidence": 0.9,
            "keywords_added": ["Python"],
            "bullet_id": "experience.0.responsibilities.0",
        },
        {
            "original": "Wrote technical notes",
            "rewritten": "Authored technical documentation adopted company-wide.",
            "confidence": 0.8,
            "keywords_added": [],
            "bullet_id": "experience.0.responsibilities.1",
        },
    ],
    "updated_summary": "",
    "keywords_inserted": ["Python"],
}


def test_approve_rewrite_keep_original_decision_reverts_one_bullet(graph, patched_tools):
    """Approving with a keep_original decision on one bullet must leave that
    bullet untouched while the other bullets rewrite still lands."""
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-keep-original"}}
    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(TWO_BULLET_REWRITTEN)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": TWO_BULLET_CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection
        graph.invoke(
            Command(resume={
                "action": "approve",
                "decisions": {
                    "experience.0.responsibilities.0": {"choice": "keep_original"},
                },
            }),
            config=config,
        )  # approve_rewrite

    state = graph.get_state(config).values
    responsibilities = state["tailored_cv_data"]["experience"][0]["responsibilities"]
    assert responsibilities[0] == "Designed the first algorithm"
    assert responsibilities[1].startswith("Authored technical documentation")


def test_approve_rewrite_custom_text_decision_overrides_rewrite(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-custom-text"}}
    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(TWO_BULLET_REWRITTEN)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": TWO_BULLET_CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)
        graph.invoke(
            Command(resume={
                "action": "approve",
                "decisions": {
                    "experience.0.responsibilities.1": {"choice": "custom", "text": "My own wording."},
                },
            }),
            config=config,
        )

    state = graph.get_state(config).values
    responsibilities = state["tailored_cv_data"]["experience"][0]["responsibilities"]
    assert responsibilities[1] == "My own wording."


def test_edit_at_approve_rewrite_locks_accepted_bullets(graph, patched_tools):
    """Editing with an explicit accept decision on one bullet must pass that
    bullet to the next rewrite call as a locked bullet (not regenerated)."""
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-edit-locks"}}
    seen_locked: list[list[dict[str, Any]]] = []

    def _rewrite_stub(kwargs):
        seen_locked.append(json.loads(kwargs.get("locked_bullets_json", "[]")))
        return json.dumps(TWO_BULLET_REWRITTEN)

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(_rewrite_stub),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": TWO_BULLET_CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection -> step5 runs once
        graph.invoke(
            Command(resume={
                "action": "edit",
                "feedback": "Make the second bullet punchier.",
                "decisions": {
                    "experience.0.responsibilities.0": {"choice": "accept"},
                },
            }),
            config=config,
        )  # approve_rewrite -> loops back to step5, which runs again

    # Second invocation of step5 (index 1) should have received the locked bullet.
    assert len(seen_locked) == 2
    locked_ids = {b.get("bullet_id") for b in seen_locked[1]}
    assert "experience.0.responsibilities.0" in locked_ids


def test_approve_selection_exclude_decision_drops_bullet(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-exclude-selection"}}
    two_selected = {
        "selected_bullets": [
            {"section": "experience", "original_text": "Designed the first algorithm",
             "relevance_score": 0.92, "reason": "", "bullet_id": "experience.0.responsibilities.0"},
            {"section": "experience", "original_text": "Wrote technical notes",
             "relevance_score": 0.5, "reason": "", "bullet_id": "experience.0.responsibilities.1"},
        ],
        "section_order": ["experience"],
        "sections_to_emphasize": [],
        "items_to_de_emphasize": [],
    }

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fake_tool(lambda _: json.dumps(two_selected)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": TWO_BULLET_CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        graph.invoke(
            Command(resume={
                "action": "approve",
                "decisions": {
                    "experience.0.responsibilities.1": {"choice": "exclude"},
                },
            }),
            config=config,
        )  # approve_selection

    state = graph.get_state(config).values
    remaining_ids = {b["bullet_id"] for b in state["selected_content"]["selected_bullets"]}
    assert remaining_ids == {"experience.0.responsibilities.0"}


# --------------------------------------------------------------------------- #
# Regression: selection/rewrite must never silently return nothing
#
# Observed live: a weak CV/job match caused the LLM to return
# selected_bullets=[] and rewritten_bullets=[], so the entire tailoring pass
# was a no-op with no explanation - the user saw "Top bullets: 0" /
# "Bullets rewritten: 0" and the chat then falsely claimed the CV was
# "tailored well". These tests pin the code-level fallback that guarantees
# some real content survives even when the model returns nothing.
# --------------------------------------------------------------------------- #

def test_selection_falls_back_to_cv_bullets_when_model_returns_nothing(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-selection-empty-fallback"}}
    empty_selection = {
        "selected_bullets": [],
        "section_order": [],
        "sections_to_emphasize": [],
        "items_to_de_emphasize": [],
    }

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fake_tool(lambda _: json.dumps(empty_selection)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score

    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_selection"
    # A weak match must still surface the CV's own bullets, never zero.
    assert len(payload["preview"]["selected_bullets"]) > 0
    assert all(b.get("bullet_id") for b in payload["preview"]["selected_bullets"])


def test_rewrite_falls_back_to_original_text_when_model_returns_nothing(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-rewrite-empty-fallback"}}
    empty_rewrite = {"rewritten_bullets": [], "updated_summary": "", "keywords_inserted": []}

    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fake_tool(lambda _: json.dumps(empty_rewrite)),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection

    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    # The selected bullet must still show up, unchanged, rather than an empty list.
    rewrites = payload["preview"]["rewritten_bullets"]
    assert len(rewrites) > 0
    assert rewrites[0]["rewritten"] == rewrites[0]["original"]


# --------------------------------------------------------------------------- #
# A tool that reports an error routes to the step_failed gate (retry or stop)
# instead of being shown as an empty-but-valid result.
# --------------------------------------------------------------------------- #

def _fails_then(result: dict[str, Any], empty: dict[str, Any]):
    """Tool stub that reports an error on the first call and succeeds after."""
    calls = {"n": 0}

    def _invoke(_: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"error": "sequence item 0: expected str instance, dict found", **empty})
        return json.dumps(result)

    return _fake_tool(_invoke)


EMPTY_SELECTION = {"selected_bullets": [], "section_order": [], "sections_to_emphasize": [], "items_to_de_emphasize": []}
EMPTY_REWRITE = {"rewritten_bullets": [], "updated_summary": "", "keywords_inserted": []}


def test_selection_error_routes_to_step_failed_and_retry_resumes(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-selection-error-retry"}}
    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fails_then(SELECTED, EMPTY_SELECTION),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score

        payload = _interrupt_payload(result)
        assert payload["step"] == "step_failed"
        assert payload["allowed_actions"] == ["approve", "reject"]
        assert "expected str instance" in payload["preview"]["error"]
        narration = result["messages"][-1].content
        assert "Bullet selection failed" in narration
        assert "Nothing in your CV was changed" in narration

        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # retry

    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_selection"
    assert payload["preview"]["selected_bullets"][0]["original_text"] == "Designed the first algorithm"
    assert not graph.get_state(config).values.get("step_error")


def test_selection_error_reject_stops_without_files(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-selection-error-stop"}}
    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fails_then(SELECTED, EMPTY_SELECTION),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        result = graph.invoke(Command(resume={"action": "reject"}), config=config)  # step_failed

    assert "__interrupt__" not in result
    assert not result.get("generated_files")
    assert "Stopped after bullet selection failed" in result["last_narration"]


def test_rewrite_error_routes_to_step_failed_and_retry_resumes(graph, patched_tools):
    from unittest.mock import patch as _patch

    config = {"configurable": {"thread_id": "test-rewrite-error-retry"}}
    with _patch(
        "app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
        _fails_then(REWRITTEN, EMPTY_REWRITE),
    ):
        graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # present_score
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # approve_selection

        payload = _interrupt_payload(result)
        assert payload["step"] == "step_failed"
        assert payload["preview"]["failed_step"] == "Bullet rewriting"

        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # retry

    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    assert payload["preview"]["rewritten_bullets"][0]["confidence"] == 0.88


def test_chat_context_reports_step_error():
    from app.agents.orchestrator.chat_agent import _state_context_block

    block = _state_context_block({
        "cv_data": CV,
        "step_error": {"step": "jt_step5_rewrite_enhance", "label": "Bullet rewriting", "detail": "boom"},
    })
    assert "step_error: Bullet rewriting failed - boom" in block


# --------------------------------------------------------------------------- #
# confirm_eligibility: asked once when the CV is silent on a stated must-have
# --------------------------------------------------------------------------- #

GERMAN_JOB = {**JOB, "required_languages": ["German (C1)"]}
SILENT_REPORT = {
    **COMPATIBILITY_REPORT,
    "gap_analysis": {
        **COMPATIBILITY_REPORT["gap_analysis"],
        "eligibility": [{
            "requirement": "German (C1)", "kind": "language", "status": "unclear",
            "rationale": "CV lists no languages.",
        }],
    },
}


@pytest.fixture
def silent_on_german(patched_tools):
    with patch(
        "app.agents.orchestrator.nodes_job_tailoring.calculate_compatibility_score_v2",
        _fake_tool(lambda _: json.dumps(SILENT_REPORT)),
    ):
        yield


def test_unclear_eligibility_asks_before_the_score_gate(graph, silent_on_german):
    config = {"configurable": {"thread_id": "test-eligibility-answer"}}
    result = graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": GERMAN_JOB}, config=config)

    payload = _interrupt_payload(result)
    assert payload["step"] == "confirm_eligibility"
    assert payload["kind"] == "input"
    [question] = payload["preview"]["questions"]
    assert question["language"] == "German"
    assert "I'll ask you about these next." in result["messages"][-1].content

    result = graph.invoke(
        Command(resume={"action": "edit", "feedback": "German: C1", "answers": {question["id"]: "C1"}}),
        config=config,
    )
    payload = _interrupt_payload(result)
    assert payload["step"] == "present_score"
    eligibility = payload["preview"]["gap_analysis"]["eligibility"]
    assert eligibility[0]["status"] == "met"
    assert any(d["name"] == "eligibility" for d in payload["preview"]["dimensions"])

    values = graph.get_state(config).values
    assert values["cv_data"]["languages"] == [{"language": "German", "fluency": "C1"}]
    assert values["eligibility_asked"] is True
    assert "German (C1): met" in result["messages"][-1].content


def test_skipping_eligibility_continues_unchanged_and_never_asks_again(graph, silent_on_german):
    config = {"configurable": {"thread_id": "test-eligibility-skip"}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": GERMAN_JOB}, config=config)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # skip

    payload = _interrupt_payload(result)
    assert payload["step"] == "present_score"
    assert payload["preview"]["gap_analysis"]["eligibility"][0]["status"] == "unclear"
    values = graph.get_state(config).values
    assert "languages" not in values["cv_data"]
    assert route_after_score(values) == JT_GATE_PRESENT_SCORE


def test_eligibility_answer_typed_as_text_is_understood(graph, silent_on_german):
    config = {"configurable": {"thread_id": "test-eligibility-chat"}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": GERMAN_JOB}, config=config)
    result = graph.invoke(
        Command(resume={"action": "edit", "feedback": "I'm a native German speaker"}), config=config
    )
    assert _interrupt_payload(result)["step"] == "present_score"
    assert graph.get_state(config).values["cv_data"]["languages"][0]["fluency"] == "native"


def test_unreadable_text_answer_asks_again(graph, silent_on_german):
    config = {"configurable": {"thread_id": "test-eligibility-unreadable"}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": GERMAN_JOB}, config=config)
    result = graph.invoke(Command(resume={"action": "edit", "feedback": "not sure"}), config=config)
    assert _interrupt_payload(result)["step"] == "confirm_eligibility"
    assert "couldn't match" in result["messages"][-1].content
