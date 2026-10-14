"""
Run tracing: every orchestrator graph run carries the session as tags and
metadata, which reach every node and the tool calls inside them, and the
response returns the run id so a request can be matched to its trace.

A recording callback handler stands in for LangSmith; nothing leaves the
process.

Run with::

    cd backend
    pytest tests/test_run_tracing.py -v
"""

from __future__ import annotations

import uuid
from typing import Any, TypedDict
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

import app.api.orchestrator as orchestrator_module
from app.api.orchestrator import _run_config

from tests.test_orchestrator_api import enabled_client, patched_tools  # noqa: F401  (fixtures)
from tests.test_orchestrator_job_tailoring import CV, JOB


class _Recorder(BaseCallbackHandler):
    def __init__(self):
        self.chains: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self.chains.append({"run_id": run_id, "parent": parent_run_id, "name": kwargs.get("name"),
                            "tags": tags or [], "metadata": metadata or {}})

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self.tools.append({"name": serialized.get("name") if serialized else None,
                           "tags": tags or [], "metadata": metadata or {}})


def test_run_config_names_and_tags_the_run():
    run_id = uuid.uuid4()
    config = _run_config("s-1", "job_tailoring", run_id)
    assert config["configurable"] == {"thread_id": "s-1"}
    assert config["run_id"] == run_id
    assert config["run_name"] == "orchestrator.job_tailoring"
    assert "session:s-1" in config["tags"]
    assert config["metadata"]["session_id"] == "s-1"


@tool
def _echo(text: str) -> str:
    """Return the text."""
    return text


class _State(TypedDict):
    text: str


def _node_calling_a_tool(state: _State) -> dict:
    # Invoked without passing config on, as the orchestrator nodes do.
    return {"text": _echo.invoke({"text": state["text"]})}


def test_tags_reach_tool_calls_inside_nodes():
    builder = StateGraph(_State)
    builder.add_node("step", _node_calling_a_tool)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    graph = builder.compile()

    recorder = _Recorder()
    run_id = uuid.uuid4()
    graph.invoke({"text": "hi"}, config={**_run_config("s-2", "job_tailoring", run_id), "callbacks": [recorder]})

    root = recorder.chains[0]
    assert root["run_id"] == run_id and root["parent"] is None
    assert recorder.tools, "the tool call inside the node was not traced"
    for run in recorder.tools + recorder.chains:
        assert "session:s-2" in run["tags"]
        assert run["metadata"].get("session_id") == "s-2"


def test_responses_return_the_run_id_of_their_graph_run(enabled_client, patched_tools):  # noqa: F811
    recorder = _Recorder()
    real_run_config = orchestrator_module._run_config

    def recording_run_config(*args, **kwargs):
        return {**real_run_config(*args, **kwargs), "callbacks": [recorder]}

    with patch.object(orchestrator_module, "_run_config", recording_run_config):
        start = enabled_client.post(
            "/api/orchestrator/start", json={"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}
        ).json()
        session_id = start["session_id"]
        resumed = enabled_client.post("/api/orchestrator/message", json={
            "session_id": session_id, "kind": "gate_resolution", "resolution": {"action": "approve"},
        }).json()

    roots = [c for c in recorder.chains if c["parent"] is None]
    assert [str(r["run_id"]) for r in roots] == [start["run_id"], resumed["run_id"]]
    assert start["run_id"] != resumed["run_id"]
    assert all(f"session:{session_id}" in c["tags"] for c in recorder.chains)
