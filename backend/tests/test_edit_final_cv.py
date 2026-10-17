"""
Editing one line of the finished CV from the chat.

Before: once the CV PDF existed the review pane said "Ask in the chat if you want to
change something", but the chat had no way to change a line; it could only re-run whole
steps. Now the chat proposes the change at the confirm_change gate (nothing changes until
the user approves), and approving it writes the line and generates the PDF again.

Run with::

    cd backend
    pytest tests/test_edit_final_cv.py -v
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import app.api.orchestrator as orchestrator_module
from app.agents.orchestrator.chat_actions import ChatActionRefused, propose_cv_edit
from app.agents.orchestrator.chat_agent import ChatTurn, EditCvLine, _state_context_block
from app.agents.orchestrator.final_cv import final_cv, final_cv_lines
from tests.test_orchestrator_api import CV, JOB, enabled_client, patched_tools  # noqa: F401
from tests.test_orchestrator_cv_review import CANNED_REVIEWS, CV_WITH_LEADERSHIP

NEW_TEXT = "Designed and validated the first published algorithm."


def _state(session_id: str) -> dict:
    return orchestrator_module._get_graph().get_state({"configurable": {"thread_id": session_id}}).values


def _send(client, session_id, **resolution):
    return client.post(
        "/api/orchestrator/message",
        json={"session_id": session_id, "kind": "gate_resolution", "resolution": resolution},
    ).json()


def _chat_edit(client, session_id, line_id, new_text=NEW_TEXT):
    turn = ChatTurn(reply="Sure, here's the change.", call=EditCvLine(tool="edit_cv_line", line_id=line_id, new_text=new_text))
    with patch("app.api.orchestrator.interpret_chat", return_value=turn):
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "chat", "text": "change that line"},
        ).json()


def _tailoring_until_cover_letter_choice(client) -> str:
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    session_id = start["session_id"]
    for _ in range(3):  # present_score, approve_selection, approve_rewrite
        body = _send(client, session_id, action="approve")
    assert body["pending_gate"]["step"] == "cover_letter_language"
    return session_id


# --------------------------------------------------------------------------- #
# Job tailoring, through the API
# --------------------------------------------------------------------------- #

def test_a_line_of_the_finished_cv_is_changed_after_confirming(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    assert _send(client, session_id, action="choose", choice="skip")["done"] is True

    line_id, old_text = final_cv_lines(_state(session_id))[0]
    proposal = _chat_edit(client, session_id, line_id)
    gate = proposal["pending_gate"]
    assert gate["step"] == "confirm_change"
    assert f"Before: {old_text}" in gate["preview"]["what_happens"]
    assert f"After: {NEW_TEXT}" in gate["preview"]["what_happens"]
    assert final_cv_lines(_state(session_id))[0][1] == old_text  # nothing changed yet

    renders = []
    with patch("app.agents.orchestrator.nodes_job_tailoring.generate_cv_pdf", SimpleNamespace(
        invoke=lambda kw: renders.append(kw) or f"CV PDF generated successfully! The file '{kw['output_filename']}' is ready for download.",
    )):
        done = _send(client, session_id, action="approve", step="confirm_change")

    assert done["done"] is True
    assert any(n.startswith("Updated your CV:") for n in done["narrations"])
    assert len(renders) == 1 and NEW_TEXT in renders[0]["cv_json"]
    assert final_cv_lines(_state(session_id))[0] == (line_id, NEW_TEXT)
    assert [f["file_type"] for f in done["generated_files"]] == ["cv"]  # replaced, not added


def test_an_edit_while_a_gate_is_open_returns_to_that_gate(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)  # the CV PDF already exists here

    line_id, _ = final_cv_lines(_state(session_id))[0]
    assert _chat_edit(client, session_id, line_id)["pending_gate"]["step"] == "confirm_change"
    back = _send(client, session_id, action="approve", step="confirm_change")

    assert back["pending_gate"]["step"] == "cover_letter_language"
    assert final_cv_lines(_state(session_id))[0][1] == NEW_TEXT


def test_cancelling_the_edit_changes_nothing(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    line_id, old_text = final_cv_lines(_state(session_id))[0]

    _chat_edit(client, session_id, line_id)
    back = _send(client, session_id, action="reject", step="confirm_change")

    assert back["pending_gate"]["step"] == "cover_letter_language"
    assert final_cv_lines(_state(session_id))[0][1] == old_text


def test_before_the_cv_exists_the_chat_explains_where_changes_go(enabled_client, patched_tools):
    client = enabled_client
    start = client.post(
        "/api/orchestrator/start",
        json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    reply = _chat_edit(client, start["session_id"], "experience.0.responsibilities.0")
    assert reply["pending_gate"]["step"] == "present_score"
    assert "no finished CV to change yet" in reply["narration"]


# --------------------------------------------------------------------------- #
# CV review, through the API
# --------------------------------------------------------------------------- #

def test_the_reviewed_cv_can_be_edited_too(enabled_client):
    client = enabled_client
    renders = []

    def _render(kw):
        renders.append(kw)
        return f"CV PDF generated successfully! The file '{kw['output_filename']}' is ready for download."

    with patch("app.agents.orchestrator.nodes_cv_review._run_step_review",
               side_effect=lambda schema_cls, *a, **k: dict(CANNED_REVIEWS.get(schema_cls.__name__, {}))), \
         patch("app.agents.orchestrator.nodes_cv_review.generate_cv_pdf", SimpleNamespace(invoke=_render)):
        body = client.post("/api/orchestrator/start", json={"flow": "cv_review", "cv_data": CV_WITH_LEADERSHIP}).json()
        session_id = body["session_id"]
        while body["pending_gate"] is not None:
            body = _send(client, session_id, action="approve")
        assert len(renders) == 1

        line_id, _ = final_cv_lines(_state(session_id))[0]
        assert _chat_edit(client, session_id, line_id)["pending_gate"]["step"] == "confirm_change"
        done = _send(client, session_id, action="approve", step="confirm_change")

    assert done["done"] is True
    assert len(renders) == 2 and renders[1]["max_pages"] == 1  # the review's recommended length still applies
    assert final_cv_lines(_state(session_id))[0] == (line_id, NEW_TEXT)


# --------------------------------------------------------------------------- #
# The proposal and the chat's view of the CV
# --------------------------------------------------------------------------- #

FINISHED = {
    "flow": "job_tailoring",
    "generated_files": [{"filename": "ada_cv_tailored.pdf", "file_type": "cv", "download_url": "/x"}],
    "tailored_cv_data": {"name": "Ada", "experience": [{"company": "AE", "responsibilities": ["Old line"]}]},
}


def test_the_final_cv_is_the_one_the_pdf_was_made_from():
    assert final_cv(FINISHED) == ("tailored_cv_data", FINISHED["tailored_cv_data"])
    assert final_cv({**FINISHED, "generated_files": []}) is None
    reviewed = {"flow": "cv_review", "generated_files": FINISHED["generated_files"], "cv_data": {"summary": "Hi there."}}
    assert final_cv_lines(reviewed) == [("summary", "Hi there.")]


def test_the_proposal_carries_the_change_without_applying_it():
    current = copy.deepcopy(FINISHED)
    proposal = propose_cv_edit(current, pending_node=None, pending_step=None,
                               line_id="experience.0.responsibilities.0", new_text="New line")
    assert proposal["restore"]["tailored_cv_data"]["experience"][0]["responsibilities"] == ["New line"]
    assert proposal["details"][:2] == ["Before: Old line", "After: New line"]
    assert current == FINISHED


@pytest.mark.parametrize("line_id, new_text, pending_step, reason", [
    ("experience.9.responsibilities.0", "New", None, "couldn't find that line"),
    ("experience.0.responsibilities.0", "  ", None, "What should the line say"),
    ("experience.0.responsibilities.0", "Old line", None, "already what the line says"),
    ("experience.0.responsibilities.0", "New", "confirm_change", "already a change waiting"),
])
def test_edits_that_cant_apply_are_refused_with_a_reason(line_id, new_text, pending_step, reason):
    with pytest.raises(ChatActionRefused, match=reason):
        propose_cv_edit(FINISHED, pending_node=None, pending_step=pending_step, line_id=line_id, new_text=new_text)


def test_the_chat_sees_the_final_cv_lines_only_once_the_cv_exists():
    assert "[experience.0.responsibilities.0] Old line" in _state_context_block(FINISHED)
    assert "Final CV lines" not in _state_context_block({**FINISHED, "generated_files": []})
