"""
Retention for session data: a session's checkpoints (CV, job, scores,
rewrites) and its generated files are deleted together once it has been
inactive for the configured TTL (SESSION_TTL_HOURS).
"""

from __future__ import annotations

from datetime import timedelta

from app.services.file_store import get_file_store
from app.services.graph_checkpointer import expire_threads, get_checkpointer


def expire_sessions(ttl: timedelta) -> list[str]:
    """Delete inactive sessions and their files; return the deleted session ids."""
    saver = get_checkpointer()
    expired = expire_threads(saver, ttl)
    store = get_file_store()
    for session_id in expired:
        store.delete_session(session_id)

    def is_live(session_id: str) -> bool:
        return saver.get_tuple({"configurable": {"thread_id": session_id}}) is not None

    # Files whose session is already gone: from before this sweep existed, or
    # written by the stateless /api/writer endpoints (no session at all).
    store.expire(ttl, is_live=is_live)
    return expired
