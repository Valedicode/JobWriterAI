"""
The documents API: the finished CV and cover letter in the editor.

The editor shows each document as its PDF prints it and saves the user's own
edits: only fields the PDF prints can change, a save made on an older revision
is refused instead of overwriting, the PDF is generated again through the
graph, and the session goes back to the gate it was on.

Run with::

    cd backend
    pytest tests/test_documents_api.py -v
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services.document_edits import EditRefused, apply_cv_edits, apply_letter_edits, letter_fields
from tests.cv_view_fixtures import RICH_EN
from tests.test_edit_final_cv import _chat_edit, _send, _state, _tailoring_until_cover_letter_choice
from tests.test_final_documents import _finish_with_cover_letter
from tests.test_orchestrator_api import enabled_client, patched_tools  # noqa: F401
from tests.test_orchestrator_cv_review import CANNED_REVIEWS, CV_WITH_LEADERSHIP

BULLETS = "experience.0.responsibilities"


def _documents(client, session_id) -> dict:
    return client.get(f"/api/documents/{session_id}").json()


def _save_cv(client, session_id, edits, revision=0):
    return client.put(f"/api/documents/{session_id}/cv", json={"revision": revision, "edits": edits})


def _bullets(view: dict) -> list[str]:
    experience = next(s for s in view["sections"] if s["type"] == "work")
    return experience["entries"][0]["bullets"]


def _renders(module: str = "nodes_job_tailoring", tool: str = "generate_cv_pdf"):
    calls: list[dict] = []

    def _render(kw):
        calls.append(kw)
        return f"PDF generated successfully! The file '{kw['output_filename']}' is ready for download."

    return calls, patch(f"app.agents.orchestrator.{module}.{tool}", SimpleNamespace(invoke=_render))


# --------------------------------------------------------------------------- #
# Viewing
# --------------------------------------------------------------------------- #

def test_the_documents_appear_once_their_pdfs_exist(enabled_client, patched_tools):
    client = enabled_client
    start = client.post("/api/orchestrator/start",
                        json={"flow": "job_tailoring", "cv_language": "english",
                              "cv_data": RICH_EN | {"skills": ["Python"]}, "job_data": {"job_title": "Engineer"}}).json()
    assert _documents(client, start["session_id"])["cv"] is None

    session_id = _tailoring_until_cover_letter_choice(client)
    docs = _documents(client, session_id)
    cv = docs["cv"]
    assert docs["editable"] is True and docs["cover_letter"] is None
    assert (cv["revision"], cv["rendered_revision"], cv["edited"]) == (0, 0, False)
    assert cv["download_url"] == _state(session_id)["generated_files"][0]["download_url"]
    assert cv["header"]["name_segs"] == [{"text": "Ada Lovelace", "src": ["name"]}]
    assert _bullets(cv)[0] == _state(session_id)["final_cv_document"]["cv"]["experience"][0]["responsibilities"][0]


def test_unsaved_edits_show_in_the_view_without_changing_anything(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    view = client.post(f"/api/documents/{session_id}/cv/view", json={"revision": 0, "edits": [
        {"op": "insert", "path": BULLETS, "index": 0, "value": "Led the first code review."},
    ]}).json()

    assert _bullets(view)[0] == "Led the first code review." and view["edited"] is True
    assert _documents(client, session_id)["cv"]["revision"] == 0


# --------------------------------------------------------------------------- #
# Saving the CV
# --------------------------------------------------------------------------- #

def test_saving_the_cv_generates_it_again_and_returns_to_the_open_gate(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    renders, stub = _renders()
    with stub:
        resp = _save_cv(client, session_id, [
            {"op": "set", "path": f"{BULLETS}.0", "value": "Designed the engine's first algorithm."},
            {"op": "delete", "path": f"{BULLETS}.1"},
            {"op": "insert", "path": BULLETS, "value": "Reviewed every release."},
        ])

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(renders) == 1 and "Reviewed every release." in renders[0]["cv_json"]
    assert renders[0]["section_order_json"] == '["experience", "skills", "education"]'  # settings kept
    assert body["session"]["pending_gate"]["step"] == "cover_letter_language"
    assert any(n.startswith("Updated your CV:") for n in body["session"]["narrations"])
    cv = body["documents"]["cv"]
    assert (cv["revision"], cv["rendered_revision"], cv["edited"]) == (1, 1, True)
    assert _bullets(cv) == ["Designed the engine's first algorithm.", "Reviewed every release."]


def test_a_save_on_an_older_revision_is_refused(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _chat_edit(client, session_id, f"{BULLETS}.0")
    _send(client, session_id, action="approve", step="confirm_change")  # revision 1 now

    resp = _save_cv(client, session_id, [{"op": "delete", "path": f"{BULLETS}.0"}], revision=0)
    assert resp.status_code == 409
    assert "changed since you opened it" in resp.json()["detail"]


def test_an_edit_to_a_field_the_cv_does_not_print_is_refused(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    before = copy.deepcopy(_state(session_id)["final_cv_document"])

    resp = _save_cv(client, session_id, [{"op": "set", "path": "fact_ledger", "value": "x"}])
    assert resp.status_code == 400 and "isn't a field the CV prints" in resp.json()["detail"]
    assert _state(session_id)["final_cv_document"] == before


def test_saving_waits_while_a_chat_proposal_is_open(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _chat_edit(client, session_id, f"{BULLETS}.0")  # confirm_change is pending

    assert _documents(client, session_id)["editable"] is False
    resp = _save_cv(client, session_id, [{"op": "delete", "path": f"{BULLETS}.0"}])
    assert resp.status_code == 409 and "waiting for your confirmation" in resp.json()["detail"]


def test_going_back_to_the_generated_cv(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    generated = _bullets(_documents(client, session_id)["cv"])
    _save_cv(client, session_id, [{"op": "delete", "path": f"{BULLETS}.0"}])

    body = client.post(f"/api/documents/{session_id}/cv/reset", json={"revision": 1}).json()
    cv = body["documents"]["cv"]
    assert _bullets(cv) == generated
    assert (cv["revision"], cv["edited"]) == (2, False)


def test_a_new_email_on_the_cv_updates_the_cover_letter_too(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _finish_with_cover_letter(client, session_id)

    letters, stub = _renders(tool="generate_cover_letter_pdf")
    with stub:
        body = _save_cv(client, session_id, [{"op": "set", "path": "email", "value": "ada@engines.example"}]).json()

    assert len(letters) == 1 and "ada@engines.example" in letters[0]["cv_json"]
    assert any("Updated your cover letter" in n for n in body["session"]["narrations"])
    assert body["documents"]["cover_letter"]["letter"]["email"] == "ada@engines.example"


def test_the_reviewed_cv_is_saved_with_its_recommended_length(enabled_client):
    client = enabled_client
    renders, stub = _renders("nodes_cv_review")
    with patch("app.agents.orchestrator.nodes_cv_review._run_step_review",
               side_effect=lambda schema_cls, *a, **k: dict(CANNED_REVIEWS.get(schema_cls.__name__, {}))), stub:
        body = client.post("/api/orchestrator/start", json={"flow": "cv_review", "cv_data": CV_WITH_LEADERSHIP}).json()
        session_id = body["session_id"]
        while body["pending_gate"] is not None:
            body = _send(client, session_id, action="approve")
        saved = _save_cv(client, session_id, [{"op": "set", "path": "name", "value": "Ada King"}]).json()

    assert saved["session"]["done"] is True
    assert len(renders) == 2 and renders[1]["max_pages"] == renders[0]["max_pages"] == 1
    assert saved["documents"]["cv"]["header"]["name"] == "Ada King"


# --------------------------------------------------------------------------- #
# The cover letter
# --------------------------------------------------------------------------- #

def test_saving_the_cover_letter_with_a_new_recipient(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _finish_with_cover_letter(client, session_id)
    letter = _documents(client, session_id)["cover_letter"]
    assert letter["letter"]["salutation"] == "Dear Dr. Jane Smith,"
    assert letter["fields"]["sign_off"] == "Yours sincerely"

    letters, stub = _renders(tool="generate_cover_letter_pdf")
    with stub:
        body = client.put(f"/api/documents/{session_id}/cover-letter", json={"revision": 0, "fields": {
            "recipient": "Hiring Team", "body_paragraph_3": "I also enjoy teaching.",
        }}).json()

    letter = body["documents"]["cover_letter"]
    assert len(letters) == 1 and letters[0]["recipient_info"] == "Hiring Team"
    assert letter["letter"]["salutation"] == "Dear Hiring Team,"
    assert letter["letter"]["sign_off"] == "Yours faithfully"  # follows the recipient
    assert [p["field"] for p in letter["letter"]["paragraph_fields"]][-2:] == ["body_paragraph_3", "closing_paragraph"]
    assert (letter["revision"], letter["rendered_revision"], letter["edited"]) == (1, 1, True)
    assert body["session"]["done"] is True


def test_the_cover_letter_can_go_back_to_the_generated_version(enabled_client, patched_tools):
    client = enabled_client
    session_id = _tailoring_until_cover_letter_choice(client)
    _finish_with_cover_letter(client, session_id)
    client.put(f"/api/documents/{session_id}/cover-letter",
               json={"revision": 0, "fields": {"opening_paragraph": "Hi."}})

    body = client.post(f"/api/documents/{session_id}/cover-letter/reset", json={"revision": 1}).json()
    letter = body["documents"]["cover_letter"]
    assert letter["fields"]["opening_paragraph"] == "Hello."
    assert letter["edited"] is False


# --------------------------------------------------------------------------- #
# The edits themselves
# --------------------------------------------------------------------------- #

RICH_DOC = {"cv": RICH_EN, "section_order": [], "convention": "international", "filename": "ada.pdf", "revision": 0}


def test_a_list_field_takes_comma_separated_text():
    cv = apply_cv_edits(RICH_DOC, [{"op": "set", "path": "projects.0.technologies", "value": "Python, Rust , "}])
    assert cv["projects"][0]["technologies"] == ["Python", "Rust"]
    assert RICH_EN["projects"][0]["technologies"] == ["Python", " NumPy ", ""]  # the document isn't changed


def test_a_bullet_goes_into_an_empty_list_the_cv_prints():
    cv = apply_cv_edits(RICH_DOC, [{"op": "insert", "path": "experience.2.responsibilities", "value": "Taught 4 pupils."}])
    assert cv["experience"][2]["responsibilities"] == ["Taught 4 pupils."]


@pytest.mark.parametrize("edit, reason", [
    ({"op": "set", "path": "experience.0.salary", "value": "1"}, "isn't a field the CV prints"),
    ({"op": "insert", "path": "experience.0.company", "value": "x"}, "isn't a bullet list"),
    ({"op": "insert", "path": "experience.0.responsibilities", "value": "  "}, "needs text"),
    ({"op": "delete", "path": "experience.0.responsibilities.9"}, "isn't a bullet the CV prints"),
    ({"op": "set", "path": "name", "value": "x" * 5001}, "longer than"),
])
def test_edits_the_cv_cannot_take_are_refused(edit, reason):
    with pytest.raises(EditRefused, match=reason):
        apply_cv_edits(RICH_DOC, [edit])


ENGLISH_LETTER = {"content": {"language": "english", "opening_paragraph": "Hello."}, "recipient": "Ms Weber"}
GERMAN_LETTER = {"content": {"language": "german", "betreff": "Bewerbung", "grussformel": ""}, "recipient": "Frau Weber"}


def test_a_sign_off_is_kept_only_when_it_differs_from_the_fitting_one():
    content, _ = apply_letter_edits(ENGLISH_LETTER, {"sign_off": "Best regards"})
    assert content["sign_off"] == "Best regards"
    content, _ = apply_letter_edits({**ENGLISH_LETTER, "content": content}, {"sign_off": "Yours sincerely"})
    assert "sign_off" not in content
    content, _ = apply_letter_edits(GERMAN_LETTER, {"sign_off": "Freundliche Grüße"})
    assert content["grussformel"] == "Freundliche Grüße"


def test_letter_fields_follow_the_language():
    assert "betreff" in letter_fields(GERMAN_LETTER) and "betreff" not in letter_fields(ENGLISH_LETTER)
    assert letter_fields(GERMAN_LETTER)["sign_off"] == "Mit freundlichen Grüßen"
    with pytest.raises(EditRefused, match="isn't a field of this letter"):
        apply_letter_edits(ENGLISH_LETTER, {"betreff": "Application"})
    with pytest.raises(EditRefused, match="needs a recipient"):
        apply_letter_edits(ENGLISH_LETTER, {"recipient": " "})
