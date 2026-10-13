"""
Session storage: the checkpointer chosen at startup, session expiry, and (when
TEST_DATABASE_URL points at a disposable Postgres database) sessions that
survive a restart.

Run with::

    cd backend
    pytest tests/test_session_persistence.py -v
    # with Postgres:
    TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/jobwriterai_test pytest tests/test_session_persistence.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import app.api.orchestrator as orchestrator_module
import app.services.graph_checkpointer as checkpointer_module
from app.agents.orchestrator.graph import build_graph
from app.main import app
from app.services.graph_checkpointer import expire_threads, get_checkpointer, reset_checkpointer
from tests.test_orchestrator_job_tailoring import CV, JOB, patched_tools  # noqa: F401 - fixture


@pytest.fixture(autouse=True)
def _fresh_checkpointer():
    reset_checkpointer()
    orchestrator_module._graph = None
    yield
    reset_checkpointer()
    orchestrator_module._graph = None


def _run_discovery(graph, thread_id: str) -> None:
    """The discovery flow ends immediately: a cheap way to create a thread."""
    graph.invoke({"flow": "discovery", "cv_data": CV}, config={"configurable": {"thread_id": thread_id}})


def _has_thread(graph, thread_id: str) -> bool:
    return bool(graph.get_state({"configurable": {"thread_id": thread_id}}).values)


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #

def test_expire_threads_deletes_only_stale_sessions():
    saver = InMemorySaver()
    graph = build_graph(checkpointer=saver)
    _run_discovery(graph, "old")
    _run_discovery(graph, "new")

    # Two days later, both are past a 24h TTL and get deleted.
    later = datetime.now(timezone.utc) + timedelta(days=2)
    with patch.object(checkpointer_module, "_now", return_value=later):
        stale = expire_threads(saver, timedelta(hours=24))
    assert set(stale) == {"old", "new"}

    # A session active just now is kept.

    _run_discovery(graph, "fresh")
    assert expire_threads(saver, timedelta(hours=24)) == []
    assert not _has_thread(graph, "old")
    assert _has_thread(graph, "fresh")


# --------------------------------------------------------------------------- #
# Startup wiring
# --------------------------------------------------------------------------- #

def test_without_database_url_sessions_stay_in_memory():
    with TestClient(app):
        assert isinstance(get_checkpointer(), InMemorySaver)


def test_with_database_url_the_postgres_saver_is_configured_and_closed(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    get_settings.cache_clear()
    saver, close = InMemorySaver(), MagicMock()
    with patch("app.main.postgres_checkpointer", return_value=(saver, close)) as factory:
        with TestClient(app):
            factory.assert_called_once_with("postgresql://example/db")
            assert get_checkpointer() is saver
        close.assert_called_once()


def test_unreachable_database_stops_startup(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    get_settings.cache_clear()
    with patch("app.main.postgres_checkpointer", side_effect=OSError("connection refused")):
        with pytest.raises(OSError):
            with TestClient(app):
                pass


# --------------------------------------------------------------------------- #
# Postgres (opt-in)
# --------------------------------------------------------------------------- #

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="set TEST_DATABASE_URL to a disposable Postgres database")
def test_postgres_session_survives_a_restart_and_expires(patched_tools):  # noqa: F811
    from app.services.graph_checkpointer import postgres_checkpointer

    thread_id = f"persistence-test-{datetime.now().timestamp()}"
    config = {"configurable": {"thread_id": thread_id}}

    saver, close = postgres_checkpointer(TEST_DATABASE_URL)
    try:
        graph = build_graph(checkpointer=saver)
        first = graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
        assert first["__interrupt__"][0].value["step"] == "present_score"
    finally:
        close()

    # "Restart": a new pool and a new graph against the same database.
    saver, close = postgres_checkpointer(TEST_DATABASE_URL)
    try:
        graph = build_graph(checkpointer=saver)
        resumed = graph.invoke(Command(resume={"action": "approve"}), config=config)
        assert resumed["__interrupt__"][0].value["step"] == "approve_selection"
        assert graph.get_state(config).values["cv_data"]["name"] == CV["name"]

        later = datetime.now(timezone.utc) + timedelta(days=2)
        with patch.object(checkpointer_module, "_now", return_value=later):
            assert thread_id in expire_threads(saver, timedelta(hours=24))
        assert not _has_thread(graph, thread_id)
    finally:
        saver.delete_thread(thread_id)
        close()


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="set TEST_DATABASE_URL to a disposable Postgres database")
def test_api_session_survives_an_app_restart(monkeypatch, patched_tools):  # noqa: F811
    from app.config import get_settings

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    get_settings.cache_clear()

    with TestClient(app) as client:  # startup creates the tables and opens the pool
        start = client.post(
            "/api/orchestrator/start",
            json={"flow": "job_tailoring", "cv_data": CV, "job_data": JOB},
        ).json()
    session_id = start["session_id"]
    assert start["pending_gate"]["step"] == "present_score"

    # Shutdown closed the pool; a fresh app process would start from nothing.
    reset_checkpointer()
    orchestrator_module._graph = None
    with TestClient(app) as client:
        resp = client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "gate_resolution", "resolution": {"action": "approve"}},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["pending_gate"]["step"] == "approve_selection"
        get_checkpointer().delete_thread(session_id)
