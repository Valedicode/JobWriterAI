"""
OrchestratorState - the TypedDict carried through every node of the orchestrator
graph.

This state holds three classes of data:

1. Routing inputs
   - flow                       : selects which branch the graph runs.
   - cv_data / job_data /
     company_data               : preprocessed payloads from the existing
                                  CV/Job/Company REST endpoints.

2. Workflow artifacts
   Job-tailoring branch:
   - job_summary
   - compatibility_report
   - tailoring_strategy
   - selected_content
   - rewritten_content
   - cover_letter_content
   - cover_letter_language
   - step_error                 : set when a step's tool failed; read by the
                                  step_failed gate (retry or stop).

   CV-review branch:
   - review_outputs

3. UI surface
   - generated_files            : list of {filename, file_type, download_url}
                                  rendered as download buttons by the frontend.
   - messages                   : conversation transcript. Uses LangGraph's
                                  add_messages reducer so individual nodes can
                                  append without clobbering history.
   - last_narration             : assistant text produced by the most recent
                                  node. The API layer returns this to the
                                  frontend as the next chat bubble.
   - pending_gate               : structured GatePayload (as a dict) populated
                                  when a node interrupts for user input.

All fields are total=False so nodes can return partial state updates.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# --------------------------------------------------------------------------- #
# Type aliases (kept narrow so static analysis catches typos in node code)
# --------------------------------------------------------------------------- #

FlowName = Literal["job_tailoring", "cv_review", "discovery"]

CoverLetterLanguage = Literal["english", "german"]

ReviewStepName = Literal[
    "header",
    "education",
    "experience",
    "leadership",
    "skills_projects",
    "assessment",
]


# --------------------------------------------------------------------------- #
# State definition
# --------------------------------------------------------------------------- #

class OrchestratorState(TypedDict, total=False):
    """State shared across every node of the orchestrator graph."""

    # --- routing inputs -----------------------------------------------------
    flow: FlowName
    cv_data: dict[str, Any]
    job_data: dict[str, Any] | None
    company_data: dict[str, Any] | None

    # --- job-tailoring artifacts -------------------------------------------
    bullet_index: dict[str, Any]
    job_summary: dict[str, Any] | None
    compatibility_report: dict[str, Any] | None
    tailoring_strategy: dict[str, Any] | None
    selected_content: dict[str, Any] | None
    rewritten_content: dict[str, Any] | None
    locked_rewrites: list[dict[str, Any]] | None
    cover_letter_content: dict[str, Any] | None
    cover_letter_language: CoverLetterLanguage | None
    cover_letter_recipient: str | None
    # Set by a step whose tool failed ({step, label, detail}); routes the graph
    # to the step_failed gate instead of presenting a fallback as real output.
    step_error: dict[str, Any] | None

    # --- cv-review artifacts -----------------------------------------------
    review_outputs: dict[str, dict[str, Any]]

    # --- output ------------------------------------------------------------
    generated_files: list[dict[str, Any]]

    # --- UI surface --------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    last_narration: str
    pending_gate: dict[str, Any] | None


def empty_state(flow: FlowName, cv_data: dict[str, Any], **extras: Any) -> OrchestratorState:
    """
    Build a minimal, well-formed initial state for a new graph thread.

    Used by the orchestrator API on POST /start to seed the checkpointer.
    Extras flow into the state untouched so callers can add job_data, etc.
    """
    state: OrchestratorState = {
        "flow": flow,
        "cv_data": cv_data,
        "job_data": extras.get("job_data"),
        "company_data": extras.get("company_data"),
        "review_outputs": {},
        "bullet_index": {},
        "generated_files": [],
        "messages": [],
        "last_narration": "",
        "pending_gate": None,
    }
    return state
