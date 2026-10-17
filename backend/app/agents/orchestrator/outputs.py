"""
Generated-file helpers shared by the orchestrator's flows.

Graph nodes get the session id from LangGraph's run config (thread_id ==
session_id) and write PDFs into that session's folder in the file store, so
each download URL is scoped to the session that produced it.
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

from langchain_core.runnables import RunnableConfig

from app.services.file_store import get_file_store, is_session_id


def session_id_of(config: RunnableConfig | None) -> str:
    """The orchestrator session (== LangGraph thread_id) this run belongs to."""
    return str(((config or {}).get("configurable") or {}).get("thread_id") or "")


def session_output_dir(config: RunnableConfig | None) -> str:
    """Folder for this session's generated files, or '' (legacy data/ root)
    when the run has no session id, e.g. a bare graph.invoke in a script."""
    session_id = session_id_of(config)
    if not is_session_id(session_id):
        return ""
    return str(get_file_store().session_dir(session_id))


def file_record(filename: str, file_type: str, config: RunnableConfig | None = None) -> dict[str, str]:
    """The {filename, file_type, download_url} entry shown as a download button."""
    session_id = session_id_of(config)
    download_url = (
        get_file_store().download_url(session_id, filename)
        if is_session_id(session_id)
        else f"/api/files/{filename}"
    )
    return {"filename": filename, "file_type": file_type, "download_url": download_url}


def with_file(files: list[Any], record: dict[str, str]) -> list[dict[str, str]]:
    """The session's file list with ``record`` in it: a regenerated PDF replaces
    the earlier record of the same name (one download button per file)."""
    out = [dict(f) for f in files or [] if isinstance(f, dict)]
    for i, existing in enumerate(out):
        if existing.get("filename") == record["filename"]:
            out[i] = record
            return out
    return out + [record]


class RenderedPdf(NamedTuple):
    """A PDF render: the updated file list, the file name (None when it
    failed), layout notes for the user, and the error when it failed."""
    files: list[dict[str, str]]
    filename: str | None
    notes: list[str]
    error: str | None


def rendered_pdf(
    tool_message: Any, filename: str, file_type: str, files: list[Any], config: RunnableConfig | None
) -> RenderedPdf:
    """The result of a generate_*_pdf tool call: on success the file list with
    the PDF's download button (a regenerated PDF replaces its old one)."""
    if isinstance(tool_message, str) and "Error" not in tool_message:
        match = re.search(r"The file '([^']+)' is ready for download", tool_message)
        name = match.group(1) if match else filename
        notes = [line for line in tool_message.splitlines() if line.startswith("Layout note:")]
        return RenderedPdf(with_file(files, file_record(name, file_type, config)), name, notes, None)
    return RenderedPdf([dict(f) for f in files or [] if isinstance(f, dict)], None, [], str(tool_message))


def render_cv_document(
    document: dict[str, Any], files: list[Any], config: RunnableConfig | None, generate_cv_pdf: Any
) -> RenderedPdf:
    """Generate the PDF of a stored CV document (final_documents) with its own
    render settings. ``generate_cv_pdf`` is the tool as the calling flow's
    module imported it."""
    cv = document.get("cv") or {}
    message = generate_cv_pdf.invoke({
        "cv_json": json.dumps(cv, default=str),
        "output_filename": document["filename"],
        "applicant_name": cv.get("name", "Applicant"),
        "section_order_json": json.dumps(document.get("section_order") or []),
        "output_dir": session_output_dir(config),
        "convention": document.get("convention") or "international",
        "max_pages": document.get("max_pages") or 0,
    })
    return rendered_pdf(message, document["filename"], "cv", files, config)
