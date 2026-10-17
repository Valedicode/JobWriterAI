"""
Chat agent for free-text turns inside the orchestrator.

This is the fix for the "hard-coded answer" problem: previously, any
``kind='chat'`` message sent while a gate was pending was either force-folded
into ``action='edit'`` (re-running an entire LLM step even for a question
like "what does ATS keyword density mean?") or rejected with a canned string
when the gate didn't support edits at all. There was no path for the model to
actually read a question and answer it.

``interpret_chat`` is a single structured-output LLM call that looks at the
user's message, the current gate (if any), the session and the recent chat
turns, and picks at most one tool:
  - none: a question or comment, answered with the session left as it is;
  - resolve_gate: one of the gate's ``allowed_actions`` -> ``ChatTurn.action``;
  - edit_bullet / rerun_from / update_profile -> ``ChatTurn.call``, which the
    API checks and carries out through ``chat_actions`` (a re-run or profile
    update only ever becomes a proposal at the confirm_change gate).

The explicit gate buttons in the UI remain the deterministic path (they call
the graph directly with no LLM involved). This module is only the flexible
path for free text, and it is intentionally conservative: it only ever emits
an action that is in ``allowed_actions``, and for a 'choice' gate the choice
must be one of the gate's own ``choices``. When it isn't sure, it must answer
and leave the gate pending rather than guess at an action.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from app.openai_llm import chat_model
from pydantic import BaseModel, Field, model_validator

from app.agents.orchestrator.final_cv import final_cv_lines
from app.agents.orchestrator.gates import GatePayload, GateResolution

logger = logging.getLogger(__name__)


class ResolveGate(BaseModel):
    """Answer the pending gate, like pressing one of its buttons."""

    tool: Literal["resolve_gate"]
    action: Literal["approve", "reject", "edit", "choose"]
    feedback: Optional[str] = None
    choice: Optional[str] = None


class EditBullet(BaseModel):
    """Change one rewritten bullet (only while approve_rewrite is pending)."""

    tool: Literal["edit_bullet"]
    bullet_id: str = Field(description="The bullet_id, copied exactly from the rewrites in the preview.")
    instruction: str = Field(description="What to change in that bullet, in the user's terms.")


class RerunFrom(BaseModel):
    """Go back to an earlier step and run forward again (the user confirms first)."""

    tool: Literal["rerun_from"]
    step: Literal["score", "strategy", "select", "rewrite", "cover_letter"]
    note: Optional[str] = Field(
        default=None,
        description="Guidance for the re-run (select / rewrite only), e.g. 'focus on backend work'.",
    )


class UpdateProfile(BaseModel):
    """Add a skill or spoken language the user says they have (the user confirms first)."""

    tool: Literal["update_profile"]
    kind: Literal["skill", "language"]
    value: str = Field(description="The skill or language name, e.g. 'Docker', 'German'.")
    fluency: Optional[str] = Field(default=None, description="Language level as the user stated it, e.g. 'C1'.")


class EditCvLine(BaseModel):
    """Change one line of the finished CV (the user confirms first; the PDF is generated again)."""

    tool: Literal["edit_cv_line"]
    line_id: str = Field(description="The line's id, copied exactly from 'Final CV lines'.")
    new_text: str = Field(description="The complete new wording of that line.")


ChatCall = Union[ResolveGate, EditBullet, RerunFrom, UpdateProfile, EditCvLine]


class ChatDecision(BaseModel):
    """Structured output of the chat-interpretation LLM call."""

    reply: str = Field(
        description=(
            "Conversational reply shown to the user as a chat bubble. Always "
            "populated, even when a tool is also called - briefly say what "
            "you're doing (e.g. 'Sure, approving that.')."
        )
    )
    call: Optional[ChatCall] = Field(
        default=None,
        description=(
            "At most one tool call. Leave unset for questions, comments, or "
            "anything ambiguous - answer in `reply` instead and leave the "
            "session as it is."
        ),
    )


class ChatTurn(BaseModel):
    """
    Return type of ``interpret_chat``: a reply, plus either a gate resolution
    (``action``) or one of the other tool calls (``call``) - never both.
    """

    reply: str
    action: Optional[GateResolution] = None
    call: Optional[Union[EditBullet, RerunFrom, UpdateProfile, EditCvLine]] = None

    @model_validator(mode="after")
    def _one_kind_of_action(self) -> "ChatTurn":
        if self.action is not None and self.call is not None:
            raise ValueError("a chat turn carries a gate resolution or a tool call, not both")
        return self


_SYSTEM_PROMPT = (
    "You are the conversational layer of a CV/job-tailoring assistant. The "
    "user is looking at a structured review step (a 'gate') in a chat UI, or "
    "the flow has already finished. You receive their free-text message plus "
    "the current context.\n\n"
    "Always write `reply`. Set `call` only when the message clearly asks for "
    "one of these; otherwise (questions, comments, anything ambiguous) answer "
    "only and leave `call` unset:\n"
    "- resolve_gate: the user answers the pending gate ('looks good, approve "
    "it', 'no, stop here', 'use German', 'make the rewrites shorter'). "
    "`action` MUST be one of the gate's allowed_actions; for a 'choice' gate "
    "`choice` MUST be one of its `choices`, verbatim; for 'edit', `feedback` "
    "is a short, specific instruction derived from the message. Never use it "
    "when no gate is pending.\n"
    "- edit_bullet: while step is 'approve_rewrite', the user wants ONE "
    "specific rewritten bullet changed ('make the second bullet mention the "
    "team size'). Copy its `bullet_id` exactly from the preview. For changes "
    "to all rewrites, use resolve_gate with 'edit' instead.\n"
    "- rerun_from: the user explicitly asks to redo or go back to an earlier "
    "step ('redo the bullet selection', 'start over from the score'). The "
    "user confirms before anything changes.\n"
    "- update_profile: the user states they HAVE a skill or speak a language "
    "that is missing from their CV ('I used Docker in a university course', "
    "'I speak German at C1'). Only what they explicitly claim; never infer a "
    "skill from a question like 'should I learn Docker?'. The user confirms "
    "before it is added and the score is recalculated.\n"
    "- edit_cv_line: once the CV PDF exists (the context lists 'Final CV lines'), "
    "the user wants one line of the finished CV changed ('shorten the YAMS bullet', "
    "'replace the theatre bullet with ...'). Copy the line's id exactly as `line_id` "
    "and write the complete new line in `new_text`, keeping every technology, number "
    "and date it states unless the user changes them. The user confirms before the "
    "CV is generated again. While step is 'approve_rewrite', use edit_bullet instead.\n\n"
    "Hard rules:\n"
    "- Never invent facts about the CV, job, or score that are not in the "
    "context provided. If you don't know, say so.\n"
    "- When genuinely unsure whether the user wants to act, answer only - the "
    "buttons are always there as a fallback.\n"
    "- Use the recent conversation to resolve references like 'that one' or "
    "'do it'.\n\n"
    "Honesty about fit and improvements - this is the part users complain "
    "loudest about when it's wrong:\n"
    "- Never say the CV is 'well tailored', 'looks great', or similar unless "
    "the session context actually supports it (high/excellent compatibility "
    "AND at least one bullet was actually rewritten). A medium/low score, "
    "missing_skills, or zero selected/rewritten bullets means there IS room "
    "to improve - say so plainly instead of reassuring.\n"
    "- When asked what to improve, or whether there's anything to improve, "
    "read `missing_skills`, `learnable_skills` and `transferable_skills` from the context and "
    "answer with the SPECIFIC skill names, not a generic 'looks good'. If a "
    "missing skill has no transferable bridge in the CV, say the pipeline "
    "cannot invent that experience, and suggest the user tell you if they "
    "genuinely have it (a course, a side project, brief exposure) - you can "
    "then add it to their profile and rescore.\n"
    "- If `bullets_selected` or `bullets_rewritten` is 0, say plainly that "
    "nothing was actually changed in the CV at that step, rather than implying "
    "the CV was already good. Do not guess a reason the context does not give.\n"
    "- If `step_error` is present, the last step failed: say so, say that "
    "nothing was changed, and that it can be retried. Never present a failure "
    "as the CV being fine or as nothing in it being relevant."
)


def _gate_context_block(gate: "GatePayload | Any | None") -> str:
    if gate is None:
        return "No gate is pending - the flow has already finished."
    return (
        f"step: {gate.step}\n"
        f"kind: {gate.kind}\n"
        f"allowed_actions: {gate.allowed_actions}\n"
        f"choices: {gate.choices or []}\n"
        f"gate narration: {gate.narration}\n"
        f"preview (what the user is looking at):\n{json.dumps(gate.preview, default=str)[:4000]}"
    )


def _skill_names(matches: list[Any]) -> list[str]:
    return [m.get("required_skill") for m in matches if isinstance(m, dict) and m.get("required_skill")]


def _state_context_block(state: dict[str, Any] | None) -> str:
    state = state or {}
    cv_data = state.get("cv_data") or {}
    parts = [f"Applicant: {cv_data.get('name', 'unknown')}"]
    job_summary = state.get("job_summary")
    if job_summary:
        parts.append(f"Job summary: {json.dumps(job_summary, default=str)[:1500]}")

    report = state.get("compatibility_report")
    if report:
        parts.append(
            f"Compatibility: score={report.get('aggregate_score')} level={report.get('level')} "
            f"interpretation={report.get('interpretation', '')}"
        )
        gap = report.get("gap_analysis") or {}
        missing = _skill_names(gap.get("missing_skills") or [])
        matched = _skill_names(gap.get("matched_skills") or [])

        def _via(items: list[Any]) -> list[dict[str, Any]]:
            return [
                {"required_skill": t.get("required_skill"), "via": t.get("matched_with")}
                for t in items
                if isinstance(t, dict)
            ]

        transferable = _via(gap.get("transferable_skills") or [])
        learnable = _via(gap.get("learnable_skills") or [])
        if matched:
            parts.append(f"matched_skills: {matched}")
        if transferable:
            parts.append(
                "transferable_skills (covered via a closely related skill the CV shows): "
                + json.dumps(transferable, default=str)[:1500]
            )
        if learnable:
            parts.append(
                "learnable_skills (real gaps: NOT on the CV, only adjacent experience - "
                "never describe these as covered): " + json.dumps(learnable, default=str)[:1500]
            )
        if missing:
            parts.append(
                f"missing_skills (true gaps - the CV has NOTHING to bridge these, cannot be "
                f"invented by rewriting): {missing}"
            )

    plan = (state.get("tailoring_strategy") or {}).get("plan")
    if isinstance(plan, dict):
        usable = [k.get("keyword") for k in plan.get("keyword_plan") or [] if isinstance(k, dict) and k.get("allowed")]
        parts.append(
            f"Tailoring plan: positioning={plan.get('positioning', '')!r}; headline={plan.get('headline')!r}; "
            f"terms the CV supports={usable}; red flags={plan.get('red_flags') or []}"
        )

    selected = state.get("selected_content") or {}
    rewritten = state.get("rewritten_content") or {}
    bullets_selected = len(selected.get("selected_bullets") or [])
    bullets_rewritten = len(rewritten.get("rewritten_bullets") or [])
    parts.append(f"bullets_selected: {bullets_selected}")
    parts.append(f"bullets_rewritten: {bullets_rewritten}")
    if rewritten.get("updated_summary"):
        parts.append(f"updated_summary: {rewritten['updated_summary']}")

    alignment = state.get("alignment")
    if isinstance(alignment, dict) and alignment.get("before") is not None:
        parts.append(
            f"Requirement alignment before tailoring {alignment.get('before')}, after {alignment.get('after')} "
            f"(rewrites kept {alignment.get('preservation')} of the original technologies and numbers); "
            f"now shown in a bullet: {alignment.get('newly_shown') or 'nothing new'}"
        )

    step_error = state.get("step_error")
    if isinstance(step_error, dict):
        parts.append(f"step_error: {step_error.get('label')} failed - {step_error.get('detail')}")

    confirmed = [
        f for f in (state.get("fact_ledger") or {}).values()
        if isinstance(f, dict) and f.get("source") in ("user_added", "user_confirmed")
    ]
    if confirmed:
        parts.append(
            "Confirmed by the user in this session (not on the original CV): "
            + ", ".join(str(f.get("value")) for f in confirmed)
        )

    files = state.get("generated_files") or []
    if files:
        parts.append("Generated files: " + ", ".join(f.get("filename", "") for f in files))

    lines = final_cv_lines(state)
    if lines:
        parts.append(
            "Final CV lines (the generated CV; edit_cv_line changes one of them):\n"
            + "\n".join(f"[{line_id}] {text}" for line_id, text in lines)[:6000]
        )
    return "\n".join(parts)


def _history_block(state: dict[str, Any] | None) -> str:
    turns = (state or {}).get("chat_history") or []
    lines = [
        f"{t.get('role')}: {t.get('content')}"
        for t in turns
        if isinstance(t, dict) and t.get("content")
    ]
    return "\n".join(lines) if lines else "(none)"


def interpret_chat(
    *,
    text: str,
    gate: "GatePayload | Any | None",
    state: dict[str, Any] | None = None,
) -> ChatTurn:
    """
    Interpret one free-text chat message in the context of the current gate
    (or the finished run, if ``gate is None``).

    ``gate`` accepts either the internal ``GatePayload`` or the API-layer
    ``OrchestratorGatePayload`` - they're structurally identical (same
    fields), and this function only ever reads attributes off it.

    Never raises for ordinary LLM/parsing failures - falls back to a safe
    "answer only" turn so a flaky call degrades to "couldn't process that"
    rather than crashing the request.
    """
    try:
        llm = chat_model("chat", temperature=0.2)
        structured_llm = llm.with_structured_output(ChatDecision)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("user", (
                "Current gate:\n{gate}\n\n"
                "Session context:\n{state}\n\n"
                "Recent conversation:\n{history}\n\n"
                "User message:\n{text}"
            )),
        ])
        chain = prompt | structured_llm
        decision: ChatDecision = chain.invoke({
            "gate": _gate_context_block(gate),
            "state": _state_context_block(state),
            "history": _history_block(state),
            "text": text,
        })
    except Exception:  # noqa: BLE001 - defensive fallback for any LLM/parse failure
        # The exception (provider errors, request details) goes to the log,
        # not into a chat bubble.
        logger.exception("Chat interpretation failed")
        return ChatTurn(
            reply=(
                "Sorry, I couldn't process that message just now. "
                "You can use the buttons below, or try again."
            ),
            action=None,
        )

    call = decision.call
    if call is None:
        return ChatTurn(reply=decision.reply)
    if isinstance(call, ResolveGate):
        resolution = _gate_resolution(call, gate)
        if resolution is None:
            # The model's reply already announces the action ('Sure, approving
            # that.'); nothing will happen, so it is replaced by the reason.
            return ChatTurn(reply=_unmatched_reason(call, gate))
        return ChatTurn(reply=decision.reply, action=resolution)
    # The other tools are checked against the session by the API
    # (chat_actions), which knows the checkpoint history.
    return ChatTurn(reply=decision.reply, call=call)


_ACTION_WORDS = {"approve": "approve", "reject": "reject", "edit": "give notes", "choose": "pick an option"}


def _unmatched_reason(call: ResolveGate, gate: "GatePayload | Any | None") -> str:
    """Why a gate answer the model picked can't be applied, for the user."""
    if gate is None:
        return "There's no step waiting for an answer right now, so nothing was changed."
    if call.action not in gate.allowed_actions:
        options = ", ".join(_ACTION_WORDS.get(a, a) for a in gate.allowed_actions)
        return (
            f"I can't {_ACTION_WORDS.get(call.action, call.action)} at this step, so nothing was changed. "
            f"Here you can: {options}."
        )
    if call.action == "choose":
        return (
            f"'{call.choice or ''}' isn't one of the options here ({', '.join(gate.choices or [])}), "
            "so nothing was changed."
        )
    return "Tell me what should change, and I'll pass it on as your notes for this step. Nothing was changed yet."


def _gate_resolution(call: ResolveGate, gate: "GatePayload | Any | None") -> GateResolution | None:
    """The resolution, or None when it doesn't fit the pending gate."""
    if gate is None or call.action not in gate.allowed_actions:
        return None
    if call.action == "choose" and (not gate.choices or call.choice not in gate.choices):
        return None
    if call.action == "edit" and not (call.feedback or "").strip():
        return None
    try:
        return GateResolution(action=call.action, feedback=call.feedback, choice=call.choice)
    except Exception:
        return None
