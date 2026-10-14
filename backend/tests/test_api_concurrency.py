"""
Concurrency of the orchestrator API.

- A long-running step (LLM / LaTeX work) must not stall other requests.
  Routes used to be `async def` while calling the graph synchronously, which
  blocked the event loop for every user until the step finished.
- Two requests for the same session at once must not both resume the gate.

Both tests share one TestClient (one event loop), as a real server does.

Run with::

    cd backend
    pytest tests/test_api_concurrency.py -v
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.api.orchestrator as orchestrator_module
from app.main import app
from app.services.graph_checkpointer import reset_checkpointer
from tests.test_orchestrator_job_tailoring import CV, JOB, JOB_SUMMARY, SELECTED, patched_tools  # noqa: F401


@pytest.fixture
def client():
    reset_checkpointer()
    orchestrator_module._graph = None
    with TestClient(app) as c:
        yield c
    orchestrator_module._graph = None


def _blocking_tool(result: dict, started: threading.Event, release: threading.Event):
    def _invoke(_):
        started.set()
        assert release.wait(timeout=10), "test never released the blocked tool"
        return json.dumps(result)
    return SimpleNamespace(invoke=_invoke)


def _in_thread(fn):
    box: dict = {}
    thread = threading.Thread(target=lambda: box.setdefault("value", fn()))
    thread.start()
    return thread, box


def test_a_running_step_does_not_block_other_requests(client, patched_tools):  # noqa: F811
    started, release = threading.Event(), threading.Event()
    with patch(
        "app.agents.orchestrator.nodes_job_tailoring.generate_job_summary",
        _blocking_tool(JOB_SUMMARY, started, release),
    ):
        thread, box = _in_thread(lambda: client.post(
            "/api/orchestrator/start", json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
        ))
        try:
            assert started.wait(timeout=10)
            t0 = time.monotonic()
            health = client.get("/health")
            assert health.status_code == 200
            assert time.monotonic() - t0 < 2.0, "/health waited for the running graph step"
        finally:
            release.set()
            thread.join(timeout=10)

    assert box["value"].status_code == 200
    assert box["value"].json()["pending_gate"]["step"] == "present_score"


def test_second_request_for_a_busy_session_is_refused(client, patched_tools):  # noqa: F811
    session_id = client.post(
        "/api/orchestrator/start", json={"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB},
    ).json()["session_id"]

    def approve():
        return client.post(
            "/api/orchestrator/message",
            json={"session_id": session_id, "kind": "gate_resolution", "resolution": {"action": "approve"}},
        )

    started, release = threading.Event(), threading.Event()
    with patch(
        "app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
        _blocking_tool(SELECTED, started, release),
    ):
        thread, box = _in_thread(approve)  # present_score -> strategy -> selection (blocks)
        try:
            assert started.wait(timeout=10)
            duplicate = approve()  # e.g. a double click while the first is running
            assert duplicate.status_code == 409
        finally:
            release.set()
            thread.join(timeout=10)

    first = box["value"]
    assert first.status_code == 200
    assert first.json()["pending_gate"]["step"] == "approve_selection"
    # The lock is released afterwards: the session takes the next answer normally.
    assert approve().json()["pending_gate"]["step"] == "approve_rewrite"
