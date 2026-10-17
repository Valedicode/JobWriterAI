"""
LangGraph checkpointer factory.

Mirrors the singleton + override pattern used by
``app.services.generic_session_manager.get_session_manager``: a thread-safe
lazy initialiser returns the in-process default; ``configure_checkpointer``
swaps in a different backend at application startup.

The FastAPI lifespan (``app.main``) calls ``postgres_checkpointer`` when
``DATABASE_URL`` is set, so sessions survive restarts and can be shared by
several workers. Without it the default ``InMemorySaver`` is used and every
restart loses all sessions.

thread_id == session_id throughout the codebase.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


_checkpointer: Optional[BaseCheckpointSaver] = None
_checkpointer_lock = threading.Lock()


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Return the global checkpointer instance, creating an InMemorySaver on
    first call.

    Thread-safe via double-checked locking so concurrent requests during
    startup cannot construct two checkpointers.
    """
    global _checkpointer

    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                _checkpointer = InMemorySaver()

    return _checkpointer


def configure_checkpointer(saver: BaseCheckpointSaver) -> None:
    """
    Replace the global checkpointer.

    Intended for application startup (wiring up the Postgres saver) and for
    tests that need a fresh, isolated checkpointer per case.
    """
    global _checkpointer
    with _checkpointer_lock:
        _checkpointer = saver


def reset_checkpointer() -> None:
    """Drop the global checkpointer reference. Mainly used by tests."""
    global _checkpointer
    with _checkpointer_lock:
        _checkpointer = None


# --------------------------------------------------------------------------- #
# Postgres
# --------------------------------------------------------------------------- #

def postgres_checkpointer(database_url: str, max_connections: int = 10) -> tuple[BaseCheckpointSaver, Callable[[], None]]:
    """
    Build a Postgres-backed saver on a connection pool and create its tables.

    Returns ``(saver, close)``; call ``close`` on shutdown. Raises if the
    database is unreachable - a configured but broken DATABASE_URL should
    stop startup, not silently fall back to memory.

    The connection settings are the ones the saver requires (autocommit,
    dict rows), with prepared statements disabled so the same URL also works
    through a transaction-mode pooler such as Supabase's or PgBouncer.
    """
    # Imported here so the in-memory setup needs no Postgres driver.
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=max_connections,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=10)
    try:
        saver = PostgresSaver(pool)
        saver.setup()  # creates / migrates the checkpoint tables
    except Exception:
        pool.close()
        raise
    return saver, pool.close


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stale_thread_ids_postgres(saver: BaseCheckpointSaver, cutoff: datetime) -> list[str]:
    """
    Threads whose newest checkpoint is older than ``cutoff``, read straight
    from the saver's ``checkpoints`` table. Listing through the saver API
    would load every checkpoint's full state (CVs included) just to read a
    timestamp. This couples to the table layout of langgraph-checkpoint-
    postgres (``checkpoint->>'ts'``), which the integration test covers.
    """
    with saver._cursor() as cur:  # noqa: SLF001 - the saver's own pooled cursor
        cur.execute(
            """
            SELECT thread_id
            FROM checkpoints
            GROUP BY thread_id
            HAVING max((checkpoint->>'ts')::timestamptz) < %s
            """,
            (cutoff,),
        )
        return [row["thread_id"] for row in cur.fetchall()]


def _stale_thread_ids_generic(saver: BaseCheckpointSaver, cutoff: datetime) -> list[str]:
    newest: dict[str, datetime] = {}
    for item in saver.list(None):
        thread_id = item.config["configurable"]["thread_id"]
        ts = datetime.fromisoformat(item.checkpoint["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if thread_id not in newest or ts > newest[thread_id]:
            newest[thread_id] = ts
    return [thread_id for thread_id, ts in newest.items() if ts < cutoff]


def expire_threads(saver: BaseCheckpointSaver, ttl: timedelta) -> list[str]:
    """
    Delete every session whose last checkpoint is older than ``ttl`` and
    return the deleted thread ids. This removes the stored CV and job data
    too, so it doubles as the retention rule for session state.
    """
    cutoff = _now() - ttl
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        is_postgres = isinstance(saver, PostgresSaver)
    except ImportError:
        is_postgres = False
    stale = (_stale_thread_ids_postgres if is_postgres else _stale_thread_ids_generic)(saver, cutoff)
    for thread_id in stale:
        saver.delete_thread(thread_id)
    return stale
