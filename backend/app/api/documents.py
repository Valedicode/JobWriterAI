"""
Documents API - the finished CV and cover letter, for the editor.

- GET  /api/documents/{session_id}                    : both documents' views
- POST /api/documents/{session_id}/cv/view            : the CV's view with unsaved edits applied
- PUT  /api/documents/{session_id}/cv                 : save edits; the PDF is generated again
- POST /api/documents/{session_id}/cv/reset           : back to the generated version
- POST /api/documents/{session_id}/cover-letter/view  : the letter's view with unsaved edits applied
- PUT  /api/documents/{session_id}/cover-letter       : save edits; the PDF is generated again
- POST /api/documents/{session_id}/cover-letter/reset : back to the generated version

The editor shows a document as its PDF prints it (latex_renderer.cv_view and
cover_letter_context). Every request carries the revision the editor was
showing; a document changed since (a chat edit, a save from another tab) is
refused with 409 rather than overwritten.

A save is the user's own change, in their own words, so it doesn't wait at a
gate the way a change the chat proposes does (confirm_change). It goes through
the graph like one, though: the new document is written to the state, the
re-render node generates the PDF, and the session goes back to the gate that
was pending. Saves share the orchestrator's one-request-per-session lock and
are refused while a chat proposal waits for confirmation.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, status
from langgraph.types import Command

from app.agents.orchestrator.final_documents import (
    COVER_LETTER_DOCUMENT,
    CV_DOCUMENT,
    GENERATED_COVER_LETTER,
    GENERATED_CV_DOCUMENT,
    final_cover_letter,
    final_cv_document,
    is_edited,
    reset_to_generated,
    saved,
)
from app.agents.orchestrator.nodes_cv_edit import COVER_LETTER_RERENDER, CV_RERENDER
from app.agents.orchestrator.nodes_job_tailoring import JT_GATE_CONFIRM_CHANGE, letter_cv
from app.api.orchestrator import (
    _get_graph,
    _pending_gate_from_snapshot,
    _run_config,
    _session_turn,
    _snapshot_response,
    _thread_config,
)
from app.models.schemas import (
    CoverLetterEditRequest,
    CoverLetterView,
    CvDocumentView,
    CvEditRequest,
    DocumentResetRequest,
    DocumentSaveResponse,
    DocumentsResponse,
)
from app.services.document_edits import EditRefused, apply_cv_edits, apply_letter_edits, letter_fields
from app.services.generic_session_manager import get_session_manager
from app.services.latex_renderer import cover_letter_context, cv_view

router = APIRouter(prefix="/api/documents", tags=["Documents"])


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

def _download_url(state: dict[str, Any], filename: str) -> str | None:
    for f in state.get("generated_files") or []:
        if isinstance(f, dict) and f.get("filename") == filename:
            return f.get("download_url")
    return None


def _cv_view(document: dict[str, Any], state: dict[str, Any]) -> CvDocumentView:
    view = cv_view(document["cv"], document.get("section_order") or None,
                   convention=document.get("convention") or "international")
    return CvDocumentView(
        revision=document["revision"],
        rendered_revision=document.get("rendered_revision"),
        edited=is_edited(document, state.get(GENERATED_CV_DOCUMENT)),
        filename=document["filename"],
        download_url=_download_url(state, document["filename"]),
        **view,
    )


def _letter_view(document: dict[str, Any], state: dict[str, Any]) -> CoverLetterView:
    cv = letter_cv(state)
    _, letter = cover_letter_context(
        document["content"], cv.get("name", "Applicant"), recipient_info=document["recipient"],
        cv_data=cv, job_data=state.get("job_data"),
    )
    return CoverLetterView(
        revision=document["revision"],
        rendered_revision=document.get("rendered_revision"),
        edited=is_edited(document, state.get(GENERATED_COVER_LETTER)),
        filename=document["filename"],
        download_url=_download_url(state, document["filename"]),
        language=(document["content"].get("language") or "english").lower(),
        fields=letter_fields(document),
        letter=letter,
    )


def _refusal(snapshot: Any) -> str | None:
    """Why a save can't run now, or None."""
    pending = _pending_gate_from_snapshot(snapshot)
    if pending is not None and pending.step == "confirm_change":
        return "A change is waiting for your confirmation in the chat. Confirm or cancel it first."
    paused_on_gate = any(task.interrupts for task in snapshot.tasks or [])
    if snapshot.next and not paused_on_gate:
        return "This session stopped in the middle of a step, so its documents can't be changed."
    return None


def _documents(session_id: str, snapshot: Any) -> DocumentsResponse:
    state = snapshot.values
    cv, letter = final_cv_document(state), final_cover_letter(state)
    return DocumentsResponse(
        session_id=session_id,
        cv=_cv_view(cv, state) if cv else None,
        cover_letter=_letter_view(letter, state) if letter else None,
        editable=_refusal(snapshot) is None,
    )


def _snapshot(session_id: str) -> Any:
    snapshot = _get_graph().get_state(_thread_config(session_id))
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found or expired.")
    return snapshot


def _document(state: dict[str, Any], key: str, revision: int) -> dict[str, Any]:
    """The finished document at the revision the editor was showing."""
    document = final_cv_document(state) if key == CV_DOCUMENT else final_cover_letter(state)
    if document is None:
        what = "CV" if key == CV_DOCUMENT else "cover letter"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"There's no finished {what} yet.")
    if document["revision"] != revision:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The document changed since you opened it. Reload it to see the current version.",
        )
    return document


def _edited(change: Callable[[], Any]) -> Any:
    try:
        return change()
    except EditRefused as refused:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(refused)) from refused


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def _save(
    session_id: str, key: str, revision: int, change: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> DocumentSaveResponse:
    """Write the changed document and run the graph through its re-render node,
    back to the gate that was pending (or the end)."""
    with _session_turn(session_id):
        snapshot = _snapshot(session_id)
        refusal = _refusal(snapshot)
        if refusal:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=refusal)
        state = snapshot.values
        document = _edited(lambda: change(_document(state, key, revision), state))
        if not get_session_manager().touch(session_id):
            get_session_manager().register(session_id, state.get("flow") or "unknown")

        graph, config = _get_graph(), _thread_config(session_id)
        pending_node = snapshot.next[0] if snapshot.next else None
        node = CV_RERENDER if key == CV_DOCUMENT else COVER_LETTER_RERENDER
        graph.update_state(
            config,
            Command(goto=node, update={key: document, "after_edit_goto": pending_node}),
            as_node=JT_GATE_CONFIRM_CHANGE,
        )
        run_id = uuid.uuid4()
        result = graph.invoke(None, config=_run_config(session_id, state.get("flow"), run_id))
        session = _snapshot_response(
            session_id=session_id, invoke_result=result,
            messages_before=len(state.get("messages") or []), run_id=run_id,
        )
        return DocumentSaveResponse(session=session, documents=_documents(session_id, _snapshot(session_id)))


def _reset(document: dict[str, Any], state: dict[str, Any], generated_key: str) -> dict[str, Any]:
    generated = state.get(generated_key)
    if not isinstance(generated, dict):
        raise EditRefused("There's no generated version to go back to.")
    return reset_to_generated(document, generated)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@router.get("/{session_id}", response_model=DocumentsResponse, summary="The session's finished documents")
def get_documents(session_id: str) -> DocumentsResponse:
    return _documents(session_id, _snapshot(session_id))


@router.post("/{session_id}/cv/view", response_model=CvDocumentView, summary="The CV with unsaved edits applied")
def view_cv(session_id: str, request: CvEditRequest) -> CvDocumentView:
    state = _snapshot(session_id).values
    document = _document(state, CV_DOCUMENT, request.revision)
    edits = [e.model_dump() for e in request.edits]
    return _cv_view({**document, "cv": _edited(lambda: apply_cv_edits(document, edits))}, state)


@router.put("/{session_id}/cv", response_model=DocumentSaveResponse, summary="Save edits to the CV")
def save_cv(session_id: str, request: CvEditRequest) -> DocumentSaveResponse:
    edits = [e.model_dump() for e in request.edits]
    return _save(session_id, CV_DOCUMENT, request.revision,
                 lambda document, _: saved(document, cv=apply_cv_edits(document, edits)))


@router.post("/{session_id}/cv/reset", response_model=DocumentSaveResponse, summary="Back to the generated CV")
def reset_cv(session_id: str, request: DocumentResetRequest) -> DocumentSaveResponse:
    return _save(session_id, CV_DOCUMENT, request.revision,
                 lambda document, state: _reset(document, state, GENERATED_CV_DOCUMENT))


@router.post("/{session_id}/cover-letter/view", response_model=CoverLetterView,
             summary="The cover letter with unsaved edits applied")
def view_cover_letter(session_id: str, request: CoverLetterEditRequest) -> CoverLetterView:
    state = _snapshot(session_id).values
    document = _document(state, COVER_LETTER_DOCUMENT, request.revision)
    content, recipient = _edited(lambda: apply_letter_edits(document, request.fields))
    return _letter_view({**document, "content": content, "recipient": recipient}, state)


@router.put("/{session_id}/cover-letter", response_model=DocumentSaveResponse, summary="Save edits to the cover letter")
def save_cover_letter(session_id: str, request: CoverLetterEditRequest) -> DocumentSaveResponse:
    def change(document: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        content, recipient = apply_letter_edits(document, request.fields)
        return saved(document, content=content, recipient=recipient)

    return _save(session_id, COVER_LETTER_DOCUMENT, request.revision, change)


@router.post("/{session_id}/cover-letter/reset", response_model=DocumentSaveResponse,
             summary="Back to the generated cover letter")
def reset_cover_letter(session_id: str, request: DocumentResetRequest) -> DocumentSaveResponse:
    return _save(session_id, COVER_LETTER_DOCUMENT, request.revision,
                 lambda document, state: _reset(document, state, GENERATED_COVER_LETTER))
