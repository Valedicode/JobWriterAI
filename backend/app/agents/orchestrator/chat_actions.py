"""
Chat actions beyond answering the pending gate.

The chat agent can ask for three things the gate buttons can't express:

- ``edit_bullet``: change one rewritten bullet. Only while ``approve_rewrite``
  is pending; it becomes that gate's own ``edit`` resolution with every other
  bullet locked, so it goes through the gate like a button press would.
- ``rerun_from``: go back to an earlier step and run forward again.
- ``update_profile``: add a skill or language the user says they have, then
  rescore.

The last two never change the session directly. They become a *proposal*
that is shown at the ``confirm_change`` gate; only an explicit approve there
applies it (see ``jt_gate_confirm_change``). Applying restores the state from
just before the target step, taken from the checkpoint history, and runs
forward from that step - so every gate after it is shown again. Nothing here
resumes the graph past a gate.

Redirecting a paused graph uses ``update_state`` with a ``Command(goto=...)``.
A plain-values ``update_state`` on a paused gate would silently end the run
(the gate's pending task is dropped and nothing is scheduled), so every
redirect in this module goes through ``Command``.
"""

from __future__ import annotations

import copy
from typing import Any

from langgraph.graph import END
from langgraph.types import Command

from app.agents.orchestrator.gates import BulletDecision, GateResolution
from app.agents.orchestrator.nodes_job_tailoring import (
    JT_GATE_CONFIRM_CHANGE,
    JT_GATE_COVER_LETTER,
    JT_STEP2_COMPATIBILITY,
    JT_STEP3_STRATEGY,
    JT_STEP4_SELECT,
    JT_STEP5_REWRITE,
)
from app.agents.orchestrator.state import OrchestratorState
from app.services.fact_ledger import user_added_fact, with_facts

# Messages (user + assistant) kept as short-term chat memory.
CHAT_MEMORY_MESSAGES = 12
_MEMORY_TEXT_LIMIT = 2000

RERUN_STEPS: dict[str, tuple[str, str]] = {
    "score": (JT_STEP2_COMPATIBILITY, "the compatibility score"),
    "strategy": (JT_STEP3_STRATEGY, "the tailoring strategy"),
    "select": (JT_STEP4_SELECT, "bullet selection"),
    "rewrite": (JT_STEP5_REWRITE, "bullet rewriting"),
    "cover_letter": (JT_GATE_COVER_LETTER, "the cover letter"),
}

# Kept as they are now rather than restored from before the target step: the
# transcript, the chat itself, and the user's CV with everything they have
# confirmed since (cv_data is never tailored; step 6 writes tailored_cv_data).
_CARRIED_FORWARD = {
    "flow",
    "messages",
    "chat_history",
    "chat_proposal",
    "cv_data",
    "original_cv_data",
    "fact_ledger",
    "eligibility_asked",
}

_STATE_KEYS = tuple(OrchestratorState.__annotations__)


class ChatActionRefused(Exception):
    """The requested action doesn't fit the session's current state. The
    message is shown to the user as the reason."""


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #

def _values_before(graph: Any, config: dict[str, Any], node: str) -> dict[str, Any] | None:
    """State just before ``node`` last ran in this session, or None if it
    never has. History is newest first, so this is the latest pass."""
    for snapshot in graph.get_state_history(config):
        if snapshot.next == (node,):
            return snapshot.values
    return None


def _restore_update(before: dict[str, Any]) -> dict[str, Any]:
    """Every state key back to its value before the target step, except the
    ones carried forward. Keys that didn't exist then are cleared."""
    return {key: copy.deepcopy(before.get(key)) for key in _STATE_KEYS if key not in _CARRIED_FORWARD}


def _dropped_files_note(before: dict[str, Any], current: dict[str, Any]) -> list[str]:
    kept = {f.get("filename") for f in before.get("generated_files") or [] if isinstance(f, dict)}
    dropped = [
        f.get("filename") for f in current.get("generated_files") or []
        if isinstance(f, dict) and f.get("filename") not in kept
    ]
    if not dropped:
        return []
    return ["These files leave this chat until they are generated again: " + ", ".join(dropped) + "."]


def _require_job_tailoring(current: dict[str, Any]) -> None:
    if current.get("flow") != "job_tailoring":
        raise ChatActionRefused("I can only do that in the job-tailoring flow.")


def _require_no_open_proposal(pending_step: str | None) -> None:
    if pending_step == "confirm_change":
        raise ChatActionRefused("There's already a change waiting for your confirmation. Confirm or cancel it first.")


def propose_rerun(
    graph: Any,
    config: dict[str, Any],
    current: dict[str, Any],
    *,
    pending_node: str | None,
    pending_step: str | None,
    step: str,
    note: str | None = None,
) -> dict[str, Any]:
    _require_job_tailoring(current)
    _require_no_open_proposal(pending_step)
    if step not in RERUN_STEPS:
        raise ChatActionRefused(f"I can re-run from: {', '.join(RERUN_STEPS)}.")
    target, label = RERUN_STEPS[step]
    before = _values_before(graph, config, target)
    if before is None:
        raise ChatActionRefused(f"{label.capitalize()} hasn't run yet in this session, so there's nothing to re-run.")

    restore = _restore_update(before)
    note = (note or "").strip()
    if note and target in (JT_STEP4_SELECT, JT_STEP5_REWRITE):
        strategy = dict(restore.get("tailoring_strategy") or {})
        strategy["user_feedback"] = list(strategy.get("user_feedback") or []) + [{"stage": "chat", "text": note}]
        restore["tailoring_strategy"] = strategy

    details = [f"Everything from {label} onward runs again, and you review each step again."]
    if note and target in (JT_STEP4_SELECT, JT_STEP5_REWRITE):
        details.append(f"Your note goes to that step: {note}")
    details.extend(_dropped_files_note(before, current))
    return {
        "kind": "rerun",
        "summary": f"Re-run from {label}",
        "details": details,
        "goto": target,
        "return_to": pending_node,
        "restore": restore,
        "on_approve": f"Re-running from {label}.",
    }


def _with_profile_addition(
    cv: dict[str, Any], kind: str, value: str, fluency: str | None
) -> tuple[dict[str, Any], str]:
    """The CV with one skill or language added, and a short description."""
    cv = copy.deepcopy(cv)
    if kind == "skill":
        skills = [s for s in cv.get("skills") or [] if isinstance(s, str)]
        if any(s.strip().casefold() == value.casefold() for s in skills):
            raise ChatActionRefused(f"{value} is already in your CV's skills.")
        cv["skills"] = skills + [value]
        return cv, f"Add {value} to your skills"
    if kind == "language":
        languages = [entry for entry in cv.get("languages") or [] if isinstance(entry, dict)]
        fluency = (fluency or "").strip() or None
        for entry in languages:
            if str(entry.get("language", "")).strip().casefold() == value.casefold():
                if (entry.get("fluency") or None) == fluency:
                    raise ChatActionRefused(f"{value} is already on your CV with that level.")
                entry["fluency"] = fluency
                break
        else:
            languages.append({"language": value, "fluency": fluency})
        cv["languages"] = languages
        return cv, f"Add {value}{f' ({fluency})' if fluency else ''} to your languages"
    raise ChatActionRefused("I can add a skill or a language to your profile.")


def propose_profile_update(
    graph: Any,
    config: dict[str, Any],
    current: dict[str, Any],
    *,
    pending_node: str | None,
    pending_step: str | None,
    kind: str,
    value: str,
    fluency: str | None,
    source_text: str,
) -> dict[str, Any]:
    _require_job_tailoring(current)
    _require_no_open_proposal(pending_step)
    value = (value or "").strip()
    if not value or len(value) > 80:
        raise ChatActionRefused("Tell me the skill or language in a few words.")
    before = _values_before(graph, config, JT_STEP2_COMPATIBILITY)
    if before is None:
        raise ChatActionRefused("The compatibility score hasn't run yet in this session.")

    new_cv, summary = _with_profile_addition(current.get("cv_data") or {}, kind, value, fluency)

    restore = _restore_update(before)
    restore["cv_data"] = new_cv
    restore["fact_ledger"] = with_facts(
        current.get("fact_ledger"),
        [user_added_fact(kind, value, fluency if kind == "language" else None, source_text[:500])],
    )
    details = [
        f"{summary} (you said: “{source_text[:200]}”).",
        "Your score is recalculated, and every step after it runs again for your review.",
        *_dropped_files_note(before, current),
    ]
    return {
        "kind": "update_profile",
        "summary": summary,
        "details": details,
        "goto": JT_STEP2_COMPATIBILITY,
        "return_to": pending_node,
        "restore": restore,
        "on_approve": f"{summary}: done. Recalculating your score.",
    }


def edit_bullet_resolution(gate: Any, bullet_id: str, instruction: str) -> GateResolution:
    """The approve_rewrite 'edit' that regenerates one bullet and keeps every
    other rewrite as it is."""
    if gate is None or gate.step != "approve_rewrite":
        raise ChatActionRefused(
            "I can change a single bullet while you're reviewing the rewrites. "
            "At this step, use the buttons or tell me what to do with the step as a whole."
        )
    bullets = (gate.preview or {}).get("rewritten_bullets") or []
    ids = [b.get("bullet_id") for b in bullets if isinstance(b, dict) and b.get("bullet_id")]
    if bullet_id not in ids:
        raise ChatActionRefused("I couldn't find that bullet among the rewrites under review.")
    instruction = (instruction or "").strip()
    if not instruction:
        raise ChatActionRefused("What should change in that bullet?")
    return GateResolution(
        action="edit",
        feedback=f"Change only bullet {bullet_id}: {instruction}",
        decisions={other: BulletDecision(choice="accept") for other in ids if other != bullet_id},
    )


# --------------------------------------------------------------------------- #
# Graph redirects
# --------------------------------------------------------------------------- #

def raise_confirm_gate(graph: Any, config: dict[str, Any], proposal: dict[str, Any]) -> None:
    """Point the paused (or finished) session at the confirm_change gate. The
    caller then runs the graph with ``invoke(None, ...)`` to show the gate."""
    graph.update_state(
        config,
        Command(goto=JT_GATE_CONFIRM_CHANGE, update={"chat_proposal": proposal}),
        as_node=JT_GATE_CONFIRM_CHANGE,
    )


def remember_chat_turn(graph: Any, config: dict[str, Any], user_text: str, reply: str) -> None:
    """
    Append one exchange to the session's short-term chat memory.

    Writing state while a gate is pending needs care: the write goes through
    ``Command(goto=<the pending gate>)`` and the gate is then run again, which
    re-raises the same gate (gate nodes build their payload from state only).
    A session stopped on a step rather than a gate (a crash mid-run) is left
    alone, so a chat turn never re-runs a step.
    """
    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values or snapshot.values.get("flow") != "job_tailoring":
        return
    paused_on_gate = any(task.interrupts for task in snapshot.tasks or [])
    if snapshot.next and not paused_on_gate:
        return
    history = list(snapshot.values.get("chat_history") or []) + [
        {"role": "user", "content": user_text[:_MEMORY_TEXT_LIMIT]},
        {"role": "assistant", "content": reply[:_MEMORY_TEXT_LIMIT]},
    ]
    pending_node = snapshot.next[0] if snapshot.next else None
    graph.update_state(
        config,
        Command(goto=pending_node or END, update={"chat_history": history[-CHAT_MEMORY_MESSAGES:]}),
        as_node=JT_GATE_CONFIRM_CHANGE,
    )
    if pending_node:
        graph.invoke(None, config)
