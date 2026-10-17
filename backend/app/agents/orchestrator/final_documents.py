"""
The finished documents a session's PDFs are made from: the ones the chat and
the editor change.

Step 7 of either flow stores the CV exactly as it is printed
(``final_cv_document``): job tailoring's project order and skill groups
already applied, with the render settings and the file name. Step 9 stores the
cover letter the same way (``final_cover_letter``). Each also keeps an
untouched copy (``generated_cv_document``, ``generated_cover_letter``) for
going back to the generated version.

A chat edit or an editor save replaces the final document and generates its
PDF again (nodes_cv_edit). Every save raises ``revision``; ``rendered_revision``
is the revision the PDF shows, so a failed render is visible. Before a PDF
exists there is nothing to edit this way: during the review steps changes go
through the step's own gate.
"""

from __future__ import annotations

import copy
from typing import Any

from app.services.bullet_index import index_cv_bullets

CV_DOCUMENT = "final_cv_document"
GENERATED_CV_DOCUMENT = "generated_cv_document"
COVER_LETTER_DOCUMENT = "final_cover_letter"
GENERATED_COVER_LETTER = "generated_cover_letter"

_VERSION_KEYS = ("revision", "rendered_revision", "applicant")
# The CV fields the cover letter's header prints.
_LETTER_APPLICANT_FIELDS = ("name", "email", "phone", "linkedin_url", "github_url", "portfolio_url")


def new_document(**fields: Any) -> dict[str, Any]:
    """A document as a step first stores it: revision 0, not rendered yet."""
    return {**fields, "revision": 0, "rendered_revision": None}


def saved(document: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The document with ``changes`` applied, as the next revision."""
    return {**copy.deepcopy(document), **changes, "revision": int(document.get("revision") or 0) + 1}


def rendered(document: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The document marked as the one its PDF now shows."""
    return {**document, **extra, "rendered_revision": document.get("revision")}


def _content(document: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in document.items() if k not in _VERSION_KEYS}


def reset_to_generated(document: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """The generated version again, as the next revision of ``document``."""
    return saved(document, **_content(generated))


def is_edited(document: dict[str, Any] | None, generated: dict[str, Any] | None) -> bool:
    """Whether the document differs from the version the step generated."""
    if not document or not generated:
        return False
    return _content(document) != _content(generated)


def letter_applicant(cv: dict[str, Any] | None) -> dict[str, Any]:
    """What the cover letter's header takes from the CV; when it changes, the
    letter is generated again with the CV."""
    cv = cv or {}
    return {key: cv.get(key) for key in _LETTER_APPLICANT_FIELDS}


def _has_file(state: dict[str, Any], file_type: str) -> bool:
    return any(isinstance(f, dict) and f.get("file_type") == file_type for f in state.get("generated_files") or [])


def final_cv_document(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """The stored CV document the session's CV PDF was made from, or None."""
    state = state or {}
    document = state.get(CV_DOCUMENT)
    if not _has_file(state, "cv") or not isinstance(document, dict) or not isinstance(document.get("cv"), dict):
        return None
    return document


def final_cover_letter(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """The stored cover letter the session's letter PDF was made from, or None."""
    state = state or {}
    document = state.get(COVER_LETTER_DOCUMENT)
    if not _has_file(state, "cover_letter") or not isinstance(document, dict) \
            or not isinstance(document.get("content"), dict):
        return None
    return document


def final_cv_lines(state: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Every editable line of the final CV as (line id, text), in CV order."""
    document = final_cv_document(state)
    if document is None:
        return []
    return [(line_id, ref["text"]) for line_id, ref in index_cv_bullets(document["cv"]).items()]
