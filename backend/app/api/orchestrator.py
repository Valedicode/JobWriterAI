"""
Orchestrator API - thin REST shell around the LangGraph orchestrator.

Three endpoints:
- POST /api/orchestrator/start         : create thread + run until first gate
- POST /api/orchestrator/message       : resume with a gate resolution (or chat)
- GET  /api/orchestrator/state/{id}    : debug snapshot of the current thread

Thread / session identity:
    session_id (HTTP layer) == thread_id (LangGraph checkpointer key).

A single compiled graph instance is cached at module load time and reused
across requests. Per-session isolation is provided by the checkpointer.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from app.agents.orchestrator import GateResolution, build_graph
from app.agents.orchestrator.chat_actions import (
    ChatActionRefused,
    edit_bullet_resolution,
    propose_profile_update,
    propose_rerun,
    raise_confirm_gate,
    remember_chat_turn,
)
from app.agents.orchestrator.chat_agent import EditBullet, RerunFrom, UpdateProfile, interpret_chat
from app.models.schemas import (
    GeneratedFile,
    OrchestratorGatePayload,
    OrchestratorMessageRequest,
    OrchestratorResponse,
    OrchestratorStartRequest,
    OrchestratorStateResponse,
)
from app.config import get_settings
from app.services.generic_session_manager import get_session_manager
from app.services.session_retention import expire_sessions

router = APIRouter(prefix="/api/orchestrator", tags=["Orchestrator"])
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cached graph singleton
# --------------------------------------------------------------------------- #

_graph = None


def _get_graph():
    """Lazy singleton so the graph is built at first use rather than import."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# --------------------------------------------------------------------------- #
# One request per session at a time
# --------------------------------------------------------------------------- #
#
# Routes run in a thread pool, so two requests for the same session (a double
# click, a chat message sent while a gate answer is still running) could both
# read the same pending gate and both resume it. A second request is refused
# with 409 instead of queued: it was made against a screen that is about to
# change. The lock is per process; several worker processes sharing Postgres
# would need a database-level lock (pg_advisory_lock) instead.

_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


@contextmanager
def _session_turn(session_id: str) -> Iterator[None]:
    with _session_locks_guard:
        lock = _session_locks.setdefault(session_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This session is still working on the previous request. Try again when it finishes.",
        )
    try:
        yield
    finally:
        lock.release()


def _forget_session_locks(session_ids: list[str]) -> None:
    """Drop the locks of deleted sessions so the table doesn't grow forever."""
    with _session_locks_guard:
        for session_id in session_ids:
            lock = _session_locks.get(session_id)
            if lock is not None and not lock.locked():
                del _session_locks[session_id]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _run_config(session_id: str, flow: str | None, run_id: uuid.UUID) -> dict[str, Any]:
    """Config for one graph run. Tags and metadata are inherited by every node
    and every tool or LLM call inside it, so a trace can be filtered by session;
    the run id is returned to the client to find this request's trace."""
    return {
        **_thread_config(session_id),
        "run_id": run_id,
        "run_name": f"orchestrator.{flow or 'unknown'}",
        "tags": [f"session:{session_id}", f"flow:{flow or 'unknown'}"],
        "metadata": {"session_id": session_id, "flow": flow},
    }


def _pending_gate_from_invoke_result(result: dict[str, Any]) -> OrchestratorGatePayload | None:
    """Extract the gate payload from the ``__interrupt__`` slot, if present."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    value = first.value if hasattr(first, "value") else first.get("value")  # type: ignore[union-attr]
    if not isinstance(value, dict):
        return None
    return OrchestratorGatePayload.model_validate(value)


def _files_from_state(state_values: dict[str, Any]) -> list[GeneratedFile]:
    raw = state_values.get("generated_files") or []
    return [GeneratedFile(**f) for f in raw if isinstance(f, dict)]


def _pending_gate_from_snapshot(snapshot: Any) -> OrchestratorGatePayload | None:
    """Extract the pending gate directly from a graph state snapshot's tasks."""
    if snapshot is None:
        return None
    for task in snapshot.tasks or []:
        for task_interrupt in task.interrupts or []:
            value = task_interrupt.value if hasattr(task_interrupt, "value") else None
            if isinstance(value, dict):
                try:
                    return OrchestratorGatePayload.model_validate(value)
                except Exception:
                    continue
    return None


def _check_resolution_against_gate(
    resolution: GateResolution,
    gate: OrchestratorGatePayload | None,
    expected_step: str | None = None,
) -> None:
    """
    Refuse a resolution the pending gate cannot take, before the graph is
    resumed. Gate nodes branch on the action and treat anything unexpected as
    a fallthrough (an 'edit' on present_score ends the flow), so the check has
    to happen here; a refused resolution leaves the gate exactly as it was.

    ``expected_step`` is the gate the client was answering. A mismatch means
    the answer is stale (it was meant for an earlier gate) and is refused with
    409 rather than being applied to whichever gate is pending now.
    """
    if gate is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No gate is pending for this session; there is nothing to resolve.",
        )
    if expected_step is not None and expected_step != gate.step:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This answer was for gate '{expected_step}', but '{gate.step}' is pending now.",
        )
    if resolution.action not in gate.allowed_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Action '{resolution.action}' is not allowed at gate '{gate.step}'. "
                f"Allowed: {', '.join(gate.allowed_actions)}."
            ),
        )
    if resolution.action == "choose" and resolution.choice not in (gate.choices or []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Choice '{resolution.choice}' is not valid at gate '{gate.step}'. "
                f"Valid: {', '.join(gate.choices or [])}."
            ),
        )


def _snapshot_response(
    *,
    session_id: str,
    invoke_result: dict[str, Any],
    messages_before: int = 0,
    run_id: uuid.UUID | None = None,
) -> OrchestratorResponse:
    """
    Build an OrchestratorResponse from a graph.invoke() return value.

    ``messages_before`` is the length of ``state.messages`` immediately
    before this invoke() call. Every AIMessage appended after that point -
    i.e. every step narration produced while resolving this request, not
    just the last one - is surfaced as ``narrations`` so the chat shows the
    full sequence (job summary, score, strategy, ...) instead of only the
    text attached to whichever gate the run stopped on.
    """
    graph = _get_graph()
    snapshot = graph.get_state(_thread_config(session_id))
    state_values = snapshot.values if snapshot else {}

    gate = _pending_gate_from_invoke_result(invoke_result)
    files = _files_from_state(state_values)

    all_messages = state_values.get("messages", []) or []
    new_messages = all_messages[messages_before:]
    narrations = [m.content for m in new_messages if isinstance(m, AIMessage) and m.content]

    # The gate's own narration is rendered inside the gate panel, not as a
    # separate chat bubble - gate nodes never call `_narrate`, so it is not
    # already in `narrations`. For a finished run (no gate), fall back to
    # `last_narration` if nothing was captured (e.g. an empty new-session run).
    narration = (
        gate.narration if gate is not None
        else (narrations[-1] if narrations else state_values.get("last_narration", ""))
    )

    return OrchestratorResponse(
        success=True,
        session_id=session_id,
        narration=narration,
        narrations=narrations,
        pending_gate=gate,
        generated_files=files,
        done=gate is None,
        message="ok",
        run_id=str(run_id) if run_id else None,
    )


# --------------------------------------------------------------------------- #
# POST /start
# --------------------------------------------------------------------------- #

@router.post(
    "/start",
    response_model=OrchestratorResponse,
    summary="Start an orchestrator session",
    description=(
        "Create a new graph thread and run until the first interrupt or END. "
        "Use the returned session_id with /message to resolve subsequent gates."
    ),
)
def orchestrator_start(
    request: OrchestratorStartRequest,
) -> OrchestratorResponse:
    if request.flow == "job_tailoring" and not request.job_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="flow='job_tailoring' requires job_data.",
        )

    session_id = str(uuid.uuid4())
    # Opportunistic TTL sweep: delete abandoned sessions from the checkpoint
    # store itself (the registry alone never freed their state, and after a
    # restart it doesn't know the persisted sessions at all).
    expired = expire_sessions(timedelta(hours=get_settings().session_ttl_hours))
    _forget_session_locks(expired)
    session_manager = get_session_manager()
    session_manager.cleanup_expired_sessions()
    session_manager.register(session_id, request.flow)
    graph = _get_graph()
    initial_state = {
        "flow": request.flow,
        "cv_data": request.cv_data,
        "job_data": request.job_data,
        "company_data": request.company_data,
        "review_outputs": {},
        "bullet_index": {},
        "generated_files": [],
        "messages": [],
        "last_narration": "",
        "pending_gate": None,
    }

    run_id = uuid.uuid4()
    try:
        result = graph.invoke(initial_state, config=_run_config(session_id, request.flow, run_id))
    except GraphInterrupt as exc:
        # Defensive: graph.invoke() normally returns a state with __interrupt__
        # rather than raising, but we surface it cleanly either way.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph raised an unhandled interrupt: {exc}",
        ) from exc

    return _snapshot_response(session_id=session_id, invoke_result=result, messages_before=0, run_id=run_id)


# --------------------------------------------------------------------------- #
# POST /message
# --------------------------------------------------------------------------- #

@router.post(
    "/message",
    response_model=OrchestratorResponse,
    summary="Send a message to the orchestrator",
    description=(
        "Resume a paused session with a structured gate resolution, or push a "
        "free-text chat turn. A chat turn is answered in place, mapped to one "
        "of the pending gate's allowed actions, used to change one rewritten "
        "bullet (at approve_rewrite), or turned into a re-run / profile-update "
        "proposal that waits at the confirm_change gate. A resolution whose action or choice "
        "the pending gate does not allow is refused with 400; one whose `step` "
        "does not match the pending gate is refused with 409. Returns the next "
        "narration and the next pending gate (if any)."
    ),
)
def orchestrator_message(
    request: OrchestratorMessageRequest,
) -> OrchestratorResponse:
    # A plain `def` route: FastAPI runs it in its thread pool, so the graph's
    # blocking LLM and LaTeX work never stalls the event loop for other users.
    with _session_turn(request.session_id):
        return _handle_message(request)


def _handle_message(request: OrchestratorMessageRequest) -> OrchestratorResponse:
    graph = _get_graph()
    config = _thread_config(request.session_id)

    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired. Start a new session with /start.",
        )

    session_manager = get_session_manager()
    if not session_manager.touch(request.session_id):
        # A persisted session resumed after a restart: this process hasn't seen it yet.
        session_manager.register(request.session_id, snapshot.values.get("flow") or "unknown")

    if request.kind == "gate_resolution":
        if request.resolution is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="kind='gate_resolution' requires a 'resolution' object.",
            )
        # Re-validate through the orchestrator-internal model to catch shape
        # issues missed by the API-layer Pydantic types (e.g. missing feedback
        # on action='edit').
        try:
            internal = GateResolution.model_validate(request.resolution.model_dump(exclude={"step"}))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid gate resolution: {exc}",
            ) from exc
        _check_resolution_against_gate(
            internal, _pending_gate_from_snapshot(snapshot), expected_step=request.resolution.step
        )

        messages_before = len(snapshot.values.get("messages") or [])
        run_id = uuid.uuid4()
        try:
            result = graph.invoke(
                Command(resume=internal.model_dump()),
                config=_run_config(request.session_id, snapshot.values.get("flow"), run_id),
            )
        except GraphInterrupt as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Graph raised an unhandled interrupt: {exc}",
            ) from exc

        return _snapshot_response(
            session_id=request.session_id, invoke_result=result, messages_before=messages_before,
            run_id=run_id,
        )

    # kind == "chat"
    if not (request.text and request.text.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="kind='chat' requires non-empty 'text'.",
        )

    text = request.text.strip()
    pending = _pending_gate_from_snapshot(snapshot)
    turn = interpret_chat(text=text, gate=pending, state=snapshot.values)

    shown_reply = turn.reply
    try:
        if turn.action is not None:
            response = _resume_from_chat(request.session_id, snapshot, pending, turn.action, turn.reply)
        elif isinstance(turn.call, EditBullet):
            resolution = edit_bullet_resolution(pending, turn.call.bullet_id, turn.call.instruction)
            response = _resume_from_chat(request.session_id, snapshot, pending, resolution, turn.reply)
        elif isinstance(turn.call, (RerunFrom, UpdateProfile)):
            response = _propose_from_chat(request.session_id, snapshot, pending, turn.call, text, turn.reply)
        else:
            # Pure Q&A / small talk: the session stays exactly as it was.
            response = _chat_reply(request.session_id, snapshot, pending, turn.reply)
    except ChatActionRefused as refused:
        # The model asked for something the session can't do right now; its
        # reply may already promise it, so the reason replaces the reply.
        shown_reply = str(refused)
        response = _chat_reply(request.session_id, snapshot, pending, shown_reply)

    _remember(request.session_id, text, shown_reply)
    return response


def _chat_reply(
    session_id: str, snapshot: Any, pending: OrchestratorGatePayload | None, reply: str
) -> OrchestratorResponse:
    return OrchestratorResponse(
        success=True,
        session_id=session_id,
        narration=reply,
        narrations=[reply],
        pending_gate=pending,
        generated_files=_files_from_state(snapshot.values),
        done=pending is None,
        message="ok",
    )


def _resume_from_chat(
    session_id: str,
    snapshot: Any,
    pending: OrchestratorGatePayload | None,
    resolution: GateResolution,
    reply: str,
) -> OrchestratorResponse:
    """Resume the pending gate with a resolution the chat derived, checked by
    the same rule as a button press."""
    try:
        internal = GateResolution.model_validate(resolution.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chat produced an invalid gate resolution: {exc}",
        ) from exc
    _check_resolution_against_gate(internal, pending)

    graph = _get_graph()
    messages_before = len(snapshot.values.get("messages") or [])
    run_id = uuid.uuid4()
    try:
        result = graph.invoke(
            Command(resume=internal.model_dump()),
            config=_run_config(session_id, snapshot.values.get("flow"), run_id),
        )
    except GraphInterrupt as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph raised an unhandled interrupt: {exc}",
        ) from exc

    response = _snapshot_response(
        session_id=session_id, invoke_result=result, messages_before=messages_before, run_id=run_id,
    )
    if reply:
        response.narrations = [reply, *response.narrations]
    return response


def _propose_from_chat(
    session_id: str,
    snapshot: Any,
    pending: OrchestratorGatePayload | None,
    call: RerunFrom | UpdateProfile,
    text: str,
    reply: str,
) -> OrchestratorResponse:
    """Turn a re-run or profile update into a proposal at the confirm_change
    gate. Nothing in the session changes until the user approves it there."""
    graph = _get_graph()
    config = _thread_config(session_id)
    where = {
        "pending_node": snapshot.next[0] if snapshot.next else None,
        "pending_step": pending.step if pending else None,
    }
    if isinstance(call, RerunFrom):
        proposal = propose_rerun(graph, config, snapshot.values, step=call.step, note=call.note, **where)
    else:
        proposal = propose_profile_update(
            graph, config, snapshot.values,
            kind=call.kind, value=call.value, fluency=call.fluency, source_text=text, **where,
        )

    raise_confirm_gate(graph, config, proposal)
    messages_before = len(snapshot.values.get("messages") or [])
    run_id = uuid.uuid4()
    result = graph.invoke(None, config=_run_config(session_id, snapshot.values.get("flow"), run_id))
    response = _snapshot_response(
        session_id=session_id, invoke_result=result, messages_before=messages_before, run_id=run_id,
    )
    if reply:
        response.narrations = [reply, *response.narrations]
    return response


def _remember(session_id: str, text: str, reply: str) -> None:
    """Keep the exchange as short-term chat memory. A failure here must not
    lose the answer the user already got, so it is logged, not raised."""
    try:
        remember_chat_turn(_get_graph(), _thread_config(session_id), text, reply)
    except Exception:
        logger.exception("Could not store chat memory for session %s", session_id)


# --------------------------------------------------------------------------- #
# GET /state/{session_id}
# --------------------------------------------------------------------------- #

@router.get(
    "/state/{session_id}",
    response_model=OrchestratorStateResponse,
    summary="Inspect orchestrator session state",
    description="Debug-only snapshot of the latest checkpointed state for a session.",
)
def orchestrator_state(
    session_id: str,
) -> OrchestratorStateResponse:
    graph = _get_graph()
    snapshot = graph.get_state(_thread_config(session_id))
    if snapshot is None or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    state_values = snapshot.values
    # Pending gate is exposed via snapshot.tasks[].interrupts; when there are
    # no tasks left the run is done.
    pending = _pending_gate_from_snapshot(snapshot)

    return OrchestratorStateResponse(
        success=True,
        session_id=session_id,
        flow=state_values.get("flow"),
        pending_gate=pending,
        generated_files=_files_from_state(state_values),
        last_narration=state_values.get("last_narration", ""),
        done=pending is None,
    )
