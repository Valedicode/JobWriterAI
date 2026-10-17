"""
Generated files are scoped to the session that produced them.

Before: PDFs were written as data/<name>_cv_tailored.pdf and served at
/api/files/<name>, guessable from the applicant's name and shared by every
session with the same name.

Run with::

    cd backend
    pytest tests/test_session_files.py -v
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import app.api.orchestrator as orchestrator_module
from app.agents.writer_agent import _resolve_output_dir
from app.main import app
from app.services.graph_checkpointer import configure_checkpointer, reset_checkpointer
from app.services.session_retention import expire_sessions
from tests.test_orchestrator_job_tailoring import CV, JOB, patched_tools  # noqa: F401


@pytest.fixture
def client():
    reset_checkpointer()
    orchestrator_module._graph = None
    with TestClient(app) as c:
        yield c
    orchestrator_module._graph = None


def _store_pdf(store, session_id: str, name: str = "ada_cv_tailored.pdf") -> None:
    (store.session_dir(session_id) / name).write_bytes(b"%PDF-1.4 test")


# --------------------------------------------------------------------------- #
# Store and download route
# --------------------------------------------------------------------------- #

def test_store_only_resolves_safe_names_inside_the_session(_temporary_file_store):
    store = _temporary_file_store
    sid = str(uuid.uuid4())
    _store_pdf(store, sid)
    assert store.path_for(sid, "ada_cv_tailored.pdf") is not None
    assert store.path_for(sid, "../ada_cv_tailored.pdf") is None
    assert store.path_for(sid, "ada_cv_tailored.tex") is None
    assert store.path_for("not-a-session", "ada_cv_tailored.pdf") is None
    assert store.path_for(str(uuid.uuid4()), "ada_cv_tailored.pdf") is None
    with pytest.raises(ValueError):
        store.session_dir("../escape")


def test_download_needs_the_owning_session(client, _temporary_file_store):
    sid = str(uuid.uuid4())
    _store_pdf(_temporary_file_store, sid)

    ok = client.get(f"/api/files/{sid}/ada_cv_tailored.pdf")
    assert ok.status_code == 200
    assert ok.content == b"%PDF-1.4 test"
    assert client.get(f"/api/files/{uuid.uuid4()}/ada_cv_tailored.pdf").status_code == 404
    # The old name-only route is gone.
    assert client.get("/api/files/ada_cv_tailored.pdf").status_code == 404
    assert client.get(f"/api/files/{sid}/..%2F..%2Fsecret.pdf").status_code == 404


def test_a_pdf_can_be_shown_inline_for_the_editor_preview(client, _temporary_file_store):
    sid = str(uuid.uuid4())
    _store_pdf(_temporary_file_store, sid)

    inline = client.get(f"/api/files/{sid}/ada_cv_tailored.pdf?inline=true")
    assert inline.headers["content-disposition"] == 'inline; filename="ada_cv_tailored.pdf"'
    download = client.get(f"/api/files/{sid}/ada_cv_tailored.pdf")
    assert download.headers["content-disposition"] == 'attachment; filename="ada_cv_tailored.pdf"'


def test_generated_files_are_written_and_linked_per_session(client, _temporary_file_store, patched_tools):  # noqa: F811
    start = client.post(
        "/api/orchestrator/start", json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()
    sid = start["session_id"]

    def resolve(**resolution):
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": sid, "kind": "gate_resolution", "resolution": resolution},
        ).json()

    for _ in range(3):
        body = resolve(action="approve")
    assert body["pending_gate"]["step"] == "cover_letter_language"
    [cv_file] = body["generated_files"]
    assert cv_file["download_url"] == f"/api/files/{sid}/{cv_file['filename']}"
    # The PDF tool (stubbed here) was pointed at this session's folder.
    assert (_temporary_file_store.root / sid).is_dir()


def test_output_dir_must_stay_inside_data(tmp_path):
    with pytest.raises(RuntimeError):
        _resolve_output_dir(str(tmp_path))


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #

def _age(path, hours: float) -> None:
    old = time.time() - hours * 3600
    for p in [*path.iterdir(), path]:
        os.utime(p, (old, old))


def test_expiry_removes_files_of_gone_sessions_only(_temporary_file_store):
    store = _temporary_file_store
    saver = InMemorySaver()
    configure_checkpointer(saver)
    try:
        from app.agents.orchestrator.graph import build_graph
        graph = build_graph(checkpointer=saver)
        live, orphan, recent = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        graph.invoke({"flow": "discovery", "cv_data": CV}, config={"configurable": {"thread_id": live}})
        for sid in (live, orphan, recent):
            _store_pdf(store, sid)
        _age(store.root / live, 48)
        _age(store.root / orphan, 48)

        expire_sessions(timedelta(hours=24))

        assert (store.root / live).is_dir(), "an old file of a live session must stay"
        assert not (store.root / orphan).exists(), "files whose session is gone are deleted"
        assert (store.root / recent).is_dir(), "recent files are kept"
    finally:
        reset_checkpointer()


# --------------------------------------------------------------------------- #
# Output filenames can't leave the session folder
# --------------------------------------------------------------------------- #

_UNSAFE_NAMES = ["../escape.pdf", "../../app/main.pdf", "/tmp/escape.pdf", "C:/escape.pdf", "sub/dir.pdf", "cv.tex"]


@pytest.mark.parametrize("name", _UNSAFE_NAMES)
def test_pdf_tools_refuse_names_with_a_folder_part(name, _temporary_file_store):
    from app.agents.writer_agent import generate_cover_letter_pdf, generate_cv_pdf

    out = str(_temporary_file_store.session_dir(str(uuid.uuid4())))
    cv_msg = generate_cv_pdf.invoke({"cv_json": "{}", "output_filename": name, "applicant_name": "Ada", "output_dir": out})
    letter_msg = generate_cover_letter_pdf.invoke({
        "content_json": "{}", "output_filename": name, "applicant_name": "Ada", "output_dir": out,
    })
    assert cv_msg.startswith("Error:") and "filename" in cv_msg
    assert letter_msg.startswith("Error:") and "filename" in letter_msg


@pytest.mark.parametrize("name", _UNSAFE_NAMES)
def test_writer_endpoints_answer_400_for_unsafe_names(name, client):
    cv = {"name": "Ada", "email": "ada@example.com", "phone": "1", "skills": [], "education": [],
          "experience": [], "projects": []}
    plan = {"matching_experiences": [], "matching_skills": [], "relevant_projects": [],
            "keywords_to_incorporate": [], "reordering_suggestions": "", "emphasis_points": [], "reasoning": ""}
    resp = client.post(
        "/api/writer/generate-cv",
        json={"cv_data": cv, "tailoring_plan": plan, "output_filename": name},
    )
    assert resp.status_code == 400 and "output filename" in resp.json()["detail"], resp.text
    resp = client.post(
        "/api/writer/generate-cover-letter",
        json={"cv_data": cv, "job_data": JOB, "output_filename": name, "recipient_info": "Hiring Team"},
    )
    assert resp.status_code == 400 and "output filename" in resp.json()["detail"], resp.text


def test_long_applicant_names_still_give_a_safe_filename():
    from app.agents.orchestrator.nodes_job_tailoring import _filename_base
    from app.services.file_store import is_safe_filename

    base = _filename_base({"name": "Maria " * 60})
    assert is_safe_filename(f"{base}_cv_tailored.pdf")
    assert is_safe_filename(f"{base}_anschreiben.pdf")
