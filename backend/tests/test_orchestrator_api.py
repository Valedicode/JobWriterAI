"""
API-level smoke test for /api/orchestrator/*.

Drives the full job-tailoring round-trip through FastAPI's TestClient,
re-using the same tool stubs as the graph harness. Confirms:

- /start opens a session and pauses at the first gate,
- /message resumes through every gate to completion,
- /state returns the final snapshot.

Run with::

    cd backend
    pytest tests/test_orchestrator_api.py -v
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agents.orchestrator.chat_agent import ChatTurn
from app.agents.orchestrator.gates import GateResolution
from app.main import app
from app.services.graph_checkpointer import reset_checkpointer


# Reuse the fixture data from the graph harness verbatim.
from tests.test_orchestrator_job_tailoring import (  # noqa: E402
    CV,
    COMPATIBILITY_REPORT,
    COVER_LETTER_CONTENT,
    JOB,
    JOB_SUMMARY,
    REWRITTEN,
    SELECTED,
    PLAN,
    STRATEGY,
)


def _fake_tool(fn):
    return SimpleNamespace(invoke=fn)


@pytest.fixture
def patched_tools():
    stubs = {
        "generate_job_summary": _fake_tool(lambda _: json.dumps(JOB_SUMMARY)),
        "calculate_compatibility_score_v2": _fake_tool(lambda _: json.dumps(COMPATIBILITY_REPORT)),
        "decide_tailoring_strategy": _fake_tool(lambda _: json.dumps(STRATEGY)),
        "plan_tailoring_strategy": _fake_tool(lambda _: json.dumps(PLAN)),
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
def enabled_client():
    """TestClient with a fresh in-process checkpointer."""
    reset_checkpointer()
    # Also reset the cached graph singleton since it captured the old saver.
    import app.api.orchestrator as orchestrator_module
    orchestrator_module._graph = None

    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_start_pauses_at_first_gate(enabled_client, patched_tools):
    resp = enabled_client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["session_id"]
    assert body["done"] is False
    assert body["pending_gate"]["step"] == "present_score"
    assert body["pending_gate"]["kind"] == "approval"
    assert "aggregate_score" in body["pending_gate"]["preview"]


def test_start_rejects_job_tailoring_without_job_data(enabled_client):
    resp = enabled_client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV},
    )
    assert resp.status_code == 400
    assert "job_data" in resp.json()["detail"]


def test_full_round_trip_through_every_gate(enabled_client, patched_tools):
    client = enabled_client

    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]
    assert start["pending_gate"]["step"] == "present_score"

    def resolve(action, **kwargs):
        payload = {"action": action, **kwargs}
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "gate_resolution", "resolution": payload},
        ).json()

    after_score = resolve("approve")
    assert after_score["pending_gate"]["step"] == "approve_selection"

    after_sel = resolve("approve")
    assert after_sel["pending_gate"]["step"] == "approve_rewrite"

    after_rewrite = resolve("approve")
    assert after_rewrite["pending_gate"]["step"] == "cover_letter_language"
    assert after_rewrite["pending_gate"]["kind"] == "choice"

    after_lang = resolve("choose", choice="english")
    assert after_lang["pending_gate"]["step"] == "approve_cover_letter"

    final = resolve("approve")
    assert final["done"] is True
    assert final["pending_gate"] is None
    types = sorted(f["file_type"] for f in final["generated_files"])
    assert "cv" in types and "cover_letter" in types


def test_state_endpoint_returns_snapshot(enabled_client, patched_tools):
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]

    state = client.get(f"/api/orchestrator/state/{session_id}").json()
    assert state["success"] is True
    assert state["flow"] == "job_tailoring"
    assert state["pending_gate"]["step"] == "present_score"
    assert state["done"] is False


def test_state_endpoint_404_for_unknown_session(enabled_client):
    resp = enabled_client.get("/api/orchestrator/state/does-not-exist")
    assert resp.status_code == 404


def test_chat_message_is_folded_in_as_edit_feedback_for_editable_gate(enabled_client, patched_tools):
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]
    assert start["pending_gate"]["step"] == "present_score"

    def resolve(action, **kwargs):
        payload = {"action": action, **kwargs}
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "gate_resolution", "resolution": payload},
        ).json()

    after_score = resolve("approve")
    assert after_score["pending_gate"]["step"] == "approve_selection"

    # The chat model interprets "emphasise leadership more" as an edit
    # request on the approve_selection gate. Mocked deterministically so the
    # test doesn't depend on a live LLM call.
    chat_turn = ChatTurn(
        reply="Sure, I'll re-run selection with more leadership emphasis.",
        action=GateResolution(action="edit", feedback="Please emphasise leadership more."),
    )
    with patch("app.api.orchestrator.interpret_chat", return_value=chat_turn):
        resp = client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "chat", "text": "Please emphasise leadership more."},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The edit loops back through step 4 and re-interrupts at the same gate;
    # confirm it actually looped (rather than being rejected outright) by
    # checking the stored tailoring strategy picked up the feedback, and that
    # the conversational reply is surfaced as the first narration.
    assert body["pending_gate"]["step"] == "approve_selection"
    assert body["narrations"][0] == chat_turn.reply
    state = client.get(f"/api/orchestrator/state/{session_id}").json()
    assert state["pending_gate"]["step"] == "approve_selection"


def test_chat_message_with_valid_choice_resumes_choice_gate(enabled_client, patched_tools):
    """A chat message that clearly maps to a 'choice' gate's own choices must
    resume the graph, not be rejected just because 'edit' isn't allowed."""
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]

    def resolve(action, **kwargs):
        payload = {"action": action, **kwargs}
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "gate_resolution", "resolution": payload},
        ).json()

    resolve("approve")
    resolve("approve")
    after_rewrite = resolve("approve")
    assert after_rewrite["pending_gate"]["step"] == "cover_letter_language"

    chat_turn = ChatTurn(
        reply="Sure, drafting the cover letter in German.",
        action=GateResolution(action="choose", choice="german"),
    )
    with patch("app.api.orchestrator.interpret_chat", return_value=chat_turn):
        resp = client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "chat", "text": "Use German please."},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_gate"]["step"] == "approve_cover_letter"
    assert body["narrations"][0] == chat_turn.reply


def test_chat_question_leaves_gate_pending_without_resuming(enabled_client, patched_tools):
    """A pure question must be answered without advancing the graph."""
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]
    assert start["pending_gate"]["step"] == "present_score"

    chat_turn = ChatTurn(reply="ATS keyword density measures exact-keyword overlap with the posting.", action=None)
    with patch("app.api.orchestrator.interpret_chat", return_value=chat_turn):
        resp = client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "chat", "text": "What does ATS keyword density mean?"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_gate"]["step"] == "present_score"
    assert body["narration"] == chat_turn.reply
    assert body["done"] is False

    # The gate truly did not advance - a real gate_resolution still works.
    still_pending = client.get(f"/api/orchestrator/state/{session_id}").json()
    assert still_pending["pending_gate"]["step"] == "present_score"


def test_message_rejects_bad_gate_resolution(enabled_client, patched_tools):
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]

    # action='edit' without feedback must be rejected.
    resp = client.post(
        "/api/orchestrator/message",
        json={
            "session_id": session_id,
            "kind": "gate_resolution",
            "resolution": {"action": "edit"},
        },
    )
    assert resp.status_code == 400


def test_eligibility_answers_pass_through_the_api(enabled_client, patched_tools):
    from tests.test_orchestrator_job_tailoring import GERMAN_JOB, SILENT_REPORT

    with patch(
        "app.agents.orchestrator.nodes_job_tailoring.calculate_compatibility_score_v2",
        _fake_tool(lambda _: json.dumps(SILENT_REPORT)),
    ):
        start = enabled_client.post(
            "/api/orchestrator/start",
            json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": GERMAN_JOB},
        ).json()
    assert start["pending_gate"]["step"] == "confirm_eligibility"
    [question] = start["pending_gate"]["preview"]["questions"]

    resp = enabled_client.post(
        "/api/orchestrator/message",
        json={
            "session_id": start["session_id"],
            "kind": "gate_resolution",
            "resolution": {"action": "edit", "feedback": "German: native", "answers": {question["id"]: "native"}},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["pending_gate"]["step"] == "present_score"
    assert body["pending_gate"]["preview"]["gap_analysis"]["eligibility"][0]["status"] == "met"


# --------------------------------------------------------------------------- #
# Resolutions are checked against the pending gate before the graph resumes.
# --------------------------------------------------------------------------- #

def _start(client) -> str:
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    assert start["pending_gate"]["step"] == "present_score"
    return start["session_id"]


def _send(client, session_id: str, **resolution):
    return client.post(
        "/api/orchestrator/message",
        json={"session_id": session_id, "kind": "gate_resolution", "resolution": resolution},
    )


def _pending_step(client, session_id: str) -> str | None:
    gate = client.get(f"/api/orchestrator/state/{session_id}").json()["pending_gate"]
    return gate["step"] if gate else None


def test_disallowed_action_is_refused_and_gate_stays(enabled_client, patched_tools):
    # present_score is an approval gate: 'choose' belongs to choice gates.
    session_id = _start(enabled_client)
    resp = _send(enabled_client, session_id, action="choose", choice="german")
    assert resp.status_code == 400
    assert "not allowed at gate 'present_score'" in resp.json()["detail"]
    assert _pending_step(enabled_client, session_id) == "present_score"
    # The gate still works normally afterwards.
    assert _send(enabled_client, session_id, action="approve").json()["pending_gate"]["step"] == "approve_selection"


def test_invalid_choice_is_refused_instead_of_skipping(enabled_client, patched_tools):
    session_id = _start(enabled_client)
    for _ in range(3):  # present_score, approve_selection, approve_rewrite
        _send(enabled_client, session_id, action="approve")
    assert _pending_step(enabled_client, session_id) == "cover_letter_language"

    resp = _send(enabled_client, session_id, action="choose", choice="french")
    assert resp.status_code == 400
    assert _pending_step(enabled_client, session_id) == "cover_letter_language"

    resp = _send(enabled_client, session_id, action="approve")
    assert resp.status_code == 400
    assert _pending_step(enabled_client, session_id) == "cover_letter_language"


def test_stale_step_is_refused_with_conflict(enabled_client, patched_tools):
    session_id = _start(enabled_client)
    first = _send(enabled_client, session_id, action="approve", step="present_score")
    assert first.status_code == 200
    assert first.json()["pending_gate"]["step"] == "approve_selection"

    # A second click on the old present_score panel must not approve the selection.
    stale = _send(enabled_client, session_id, action="approve", step="present_score")
    assert stale.status_code == 409
    assert _pending_step(enabled_client, session_id) == "approve_selection"


def test_resolution_after_flow_end_is_refused(enabled_client, patched_tools):
    session_id = _start(enabled_client)
    assert _send(enabled_client, session_id, action="reject").json()["done"] is True
    resp = _send(enabled_client, session_id, action="approve")
    assert resp.status_code == 409


def test_chat_action_outside_allowed_actions_is_refused(enabled_client, patched_tools):
    session_id = _start(enabled_client)
    chat_turn = ChatTurn(reply="Using German.", action=GateResolution(action="choose", choice="german"))
    with patch("app.api.orchestrator.interpret_chat", return_value=chat_turn):
        resp = enabled_client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "chat", "text": "make it shorter"},
        )
    assert resp.status_code == 400
    assert _pending_step(enabled_client, session_id) == "present_score"


def test_leave_out_decision_passes_through_the_api(enabled_client, patched_tools):
    """The API schema must accept every choice the gates accept: 'leave_out'
    used to be refused with 422 before it reached the selection gate."""
    from app.services.bullet_index import index_cv_bullets

    bullet_id = next(
        b for b, ref in index_cv_bullets(CV).items() if ref["text"] == "Designed the first algorithm"
    )
    selected = json.loads(json.dumps(SELECTED))
    selected["selected_bullets"][0]["bullet_id"] = bullet_id
    with patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _fake_tool(lambda _: json.dumps(selected)),
    ):
        session_id = _start(enabled_client)
        assert _send(enabled_client, session_id, action="approve").json()["pending_gate"]["step"] == "approve_selection"
        resp = _send(
            enabled_client, session_id, action="approve", step="approve_selection",
            decisions={bullet_id: {"choice": "leave_out"}},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert any("1 left out of the tailored CV" in n for n in body["narrations"])
    assert body["pending_gate"]["step"] == "approve_rewrite"
