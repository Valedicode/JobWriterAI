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
   - original_cv_data           : the CV as extracted (frozen).
   - tailored_cv_data           : the tailored CV from step 6; cv_data
                                  itself is never overwritten.
   - fact_ledger                : facts the CV and the user state.
   - chat_proposal              : a chat-requested change awaiting the
                                  confirm_change gate.
   - chat_history               : the last few chat turns.

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
    # Rule-based directives plus "plan" (the validated StrategistPlan, or None
    # when it couldn't be drafted) and "plan_notes".
    tailoring_strategy: dict[str, Any] | None
    # The user's notes from editing the plan at the score review.
    strategy_feedback: list[str]
    selected_content: dict[str, Any] | None
    rewritten_content: dict[str, Any] | None
    locked_rewrites: list[dict[str, Any]] | None
    cover_letter_content: dict[str, Any] | None
    cover_letter_language: CoverLetterLanguage | None
    cover_letter_recipient: str | None
    # Set by a step whose tool failed ({step, label, detail}); routes the graph
    # to the step_failed gate instead of presenting a fallback as real output.
    step_error: dict[str, Any] | None
    # True once the confirm_eligibility gate has been shown, so it is asked
    # at most once per run (answered or skipped).
    eligibility_asked: bool
    # cv_data is the user's CV plus what they confirmed during the session
    # (eligibility answers, chat profile updates); it is never tailored.
    # original_cv_data is the CV as extracted, frozen at jt_entry.
    # tailored_cv_data is written by step 6 and rendered by steps 7-9.
    original_cv_data: dict[str, Any]
    tailored_cv_data: dict[str, Any] | None
    # fact_id -> FactEntry dump: what the CV and the user state
    # (app.services.fact_ledger). Rewrites may only claim these.
    fact_ledger: dict[str, dict[str, Any]]
    # The fact check of the latest rewrite pass (VerifierResult dump), and
    # which attempt the next check is (1 = first pass, reset when it passes).
    verifier_result: dict[str, Any] | None
    verify_attempt: int | None
    # Step 6: the rubric's evidence searched again in the tailored CV, and
    # {before, after, preservation, newly_shown} alignment figures.
    requirement_rubric_after: list[dict[str, Any]] | None
    alignment: dict[str, Any] | None

    # --- chat --------------------------------------------------------------
    # A change the chat proposed (re-run, profile update), waiting at the
    # confirm_change gate: {kind, summary, details, goto, return_to, restore}.
    chat_proposal: dict[str, Any] | None
    # Short-term chat memory: the last few {role, content} turns.
    chat_history: list[dict[str, str]]

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
