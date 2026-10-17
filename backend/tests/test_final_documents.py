"""
The finished documents a session's PDFs are made from.

Before: a chat edit wrote into tailored_cv_data, and every render applied the
plan's project order and skill groups to it again, so an edited skill was
replaced by the plan's groups and nothing recorded the CV as it was printed.
Now step 7 stores the CV exactly as printed (with its render settings and file
name, plus a generated copy for going back), step 9 stores the cover letter the
same way, and changes are made to those documents.

Run with::

    cd backend
    pytest tests/test_final_documents.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.agents.orchestrator.final_documents import (
    is_edited,
    letter_applicant,
    new_document,
    rendered,
    reset_to_generated,
    saved,
)
from app.agents.orchestrator.chat_agent import ChatTurn, RerunFrom
from tests.test_edit_final_cv import _chat_edit, _send, _state, _tailoring_until_cover_letter_choice
from tests.test_orchestrator_api import enabled_client, patched_tools  # noqa: F401

NEW_TEXT = "Designed and validated the first published algorithm."


def _finish_with_cover_letter(client, session_id) -> dict:
    """English letter to the recipient the job posting names, approved as drafted."""
    _send(client, session_id, action="choose", choice="english", step="cover_letter_language")
    return _send(client, session_id, action="approve", step="approve_cover_letter")


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #

def test_a_save_raises_the_revision_and_a_render_records_it():
    doc = new_document(cv={"name": "Ada"}, filename="ada.pdf")
    assert (doc["revision"], doc["rendered_revision"]) == (0, None)
    changed = saved(rendered(doc), cv={"name": "Ada L."})
    assert (changed["revision"], changed["rendered_revision"]) == (1, 0)  # saved, PDF not made yet
    assert rendered(changed)["rendered_revision"] == 1


def test_going_back_restores_the_generated_content_as_a_new_revision():
    generated = new_document(cv={"name": "Ada"}, filename="ada.pdf")
    edited = saved(rendered(generated), cv={"name": "Ada L."})
    assert is_edited(edited, generated)
    back = reset_to_generated(edited, generated)
    assert back["cv"] == {"name": "Ada"} and back["revision"] == 2
    assert not is_edited(back, generated)


# --------------------------------------------------------------------------- #
# Through the flow
# --------------------------------------------------------------------------- #

def test_step_7_stores_the_cv_as_printed_with_a_generated_copy(enabled_client, patched_tools):
    session_id = _tailoring_until_cover_letter_choice(enabled_client)
    state = _state(session_id)
    document = state["final_cv_document"]

    assert document["filename"] == state["generated_files"][0]["filename"]
    assert document["section_order"] == ["experience", "skills", "education"]
    assert (document["revision"], document["rendered_revision"]) == (0, 0)
    assert document["cv"]["experience"] == state["tailored_cv_data"]["experience"]
    assert not is_edited(document, state["generated_cv_document"])


def test_a_chat_edit_changes_the_stored_cv_and_its_pdf(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    line_id = "experience.0.responsibilities.0"
    _chat_edit(client, session_id, line_id)
    _send(client, session_id, action="approve", step="confirm_change")

    state = _state(session_id)
    document = state["final_cv_document"]
    assert document["cv"]["experience"][0]["responsibilities"][0] == NEW_TEXT
    assert (document["revision"], document["rendered_revision"]) == (1, 1)
    assert is_edited(document, state["generated_cv_document"])


def test_step_9_stores_the_cover_letter(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    done = _finish_with_cover_letter(client, session_id)

    state = _state(session_id)
    letter = state["final_cover_letter"]
    assert done["done"] is True
    assert letter["filename"] in {f["filename"] for f in done["generated_files"]}
    assert letter["recipient"] == "Dr. Jane Smith"  # named in the job posting
    assert letter["content"]["opening_paragraph"]
    assert letter["applicant"] == letter_applicant(state["final_cv_document"]["cv"])
    assert (letter["revision"], letter["rendered_revision"]) == (0, 0)


def test_a_bullet_edit_leaves_the_cover_letter_alone(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _finish_with_cover_letter(client, session_id)

    letters = []
    with patch("app.agents.orchestrator.nodes_job_tailoring.generate_cover_letter_pdf",
               SimpleNamespace(invoke=lambda kw: letters.append(kw) or "Error: should not run")):
        _chat_edit(client, session_id, "experience.0.responsibilities.0")
        done = _send(client, session_id, action="approve", step="confirm_change")

    assert letters == []  # the letter prints only the CV's name and contact details
    assert sorted(f["file_type"] for f in done["generated_files"]) == ["cover_letter", "cv"]


def test_rerunning_the_cover_letter_keeps_the_changes_to_the_cv(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _chat_edit(client, session_id, "experience.0.responsibilities.0")
    _send(client, session_id, action="approve", step="confirm_change")
    _finish_with_cover_letter(client, session_id)

    turn = ChatTurn(reply="Sure.", call=RerunFrom(tool="rerun_from", step="cover_letter"))
    with patch("app.api.orchestrator.interpret_chat", return_value=turn):
        client.post("/api/orchestrator/message",
                    json={"session_id": session_id, "kind": "chat", "text": "redo the cover letter"})
    back = _send(client, session_id, action="approve", step="confirm_change")

    assert back["pending_gate"]["step"] == "cover_letter_language"
    document = _state(session_id)["final_cv_document"]
    assert document["cv"]["experience"][0]["responsibilities"][0] == NEW_TEXT
    assert document["revision"] == 1
