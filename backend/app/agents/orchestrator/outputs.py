"""
Generated-file helpers shared by the orchestrator's flows.

Graph nodes get the session id from LangGraph's run config (thread_id ==
session_id) and write PDFs into that session's folder in the file store, so
each download URL is scoped to the session that produced it.
"""

from __future__ import annotations

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


class RenderedCv(NamedTuple):
    """A CV PDF render: the updated file list, the file name (None when it
    failed), layout notes for the user, and the error when it failed."""
    files: list[dict[str, str]]
    filename: str | None
    notes: list[str]
    error: str | None
