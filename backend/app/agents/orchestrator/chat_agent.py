"""
Chat agent for free-text turns inside the orchestrator.

This is the fix for the "hard-coded answer" problem: previously, any
``kind='chat'`` message sent while a gate was pending was either force-folded
into ``action='edit'`` (re-running an entire LLM step even for a question
like "what does ATS keyword density mean?") or rejected with a canned string
when the gate didn't support edits at all. There was no path for the model to
actually read a question and answer it.

``interpret_chat`` is a single structured-output LLM call that looks at the
user's message plus the current gate (if any) and decides:
  - Is this a question / comment that should just be answered, with the gate
    left exactly as it is? -> ``action=None``.
  - Or does it map to one of the gate's ``allowed_actions``? -> return a
    ``GateResolution`` the caller resumes the graph with.

The explicit gate buttons in the UI remain the deterministic path (they call
the graph directly with no LLM involved). This module is only the flexible
path for free text, and it is intentionally conservative: it only ever emits
an action that is in ``allowed_actions``, and for a 'choice' gate the choice
must be one of the gate's own ``choices``. When it isn't sure, it must answer
and leave the gate pending rather than guess at an action.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from app.openai_llm import chat_model
from pydantic import BaseModel, Field, model_validator

from app.agents.orchestrator.gates import GatePayload, GateResolution


class _ChatAction(BaseModel):
    """Mirrors GateResolution but keeps every field optional so structured
    output never fails validation on a partial/omitted action."""

    action: Literal["approve", "reject", "edit", "choose"]
    feedback: Optional[str] = None
    choice: Optional[str] = None


class ChatDecision(BaseModel):
    """Structured output of the chat-interpretation LLM call."""

    reply: str = Field(
        description=(
            "Conversational reply shown to the user as a chat bubble. Always "
            "populated, even when an action is also taken - briefly say what "
            "you're doing (e.g. 'Sure, approving that.')."
        )
    )
    action: Optional[_ChatAction] = Field(
        default=None,
        description=(
            "Set only when the message clearly maps to one of the gate's "
            "allowed_actions. Leave unset for questions, comments, or "
            "anything ambiguous - answer in `reply` instead and let the gate "
            "stay pending."
        ),
    )


class ChatTurn(BaseModel):
    """Return type of ``interpret_chat``: a reply plus an optional resolution."""

    reply: str
    action: Optional[GateResolution] = None

    @model_validator(mode="after")
    def _validate(self) -> "ChatTurn":
        return self


_SYSTEM_PROMPT = (
    "You are the conversational layer of a CV/job-tailoring assistant. The "
    "user is looking at a structured review step (a 'gate') in a chat UI, or "
    "the flow has already finished. You receive their free-text message plus "
    "the current context.\n\n"
    "Decide one of two things:\n"
    "1. ANSWER ONLY - the message is a question, comment, or anything that "
    "doesn't clearly ask you to act (e.g. 'what does ATS keyword density "
    "mean?', 'why is my score low?'). Reply helpfully using the provided "
    "context. Leave `action` unset.\n"
    "2. ANSWER + ACT - the message clearly requests one of the gate's "
    "allowed_actions (e.g. 'looks good, approve it', 'no, reject this', "
    "'use German', 'change X to Y'). Set `action` accordingly AND write a "
    "short `reply` acknowledging what you're doing.\n\n"
    "Hard rules:\n"
    "- `action.action` MUST be one of the gate's allowed_actions. If there is "
    "no pending gate, never set `action`.\n"
    "- For a 'choice' gate, `action.choice` MUST be one of the gate's "
    "`choices`, verbatim.\n"
    "- For `action.action='edit'`, `action.feedback` must be a short, "
    "specific rewrite instruction derived from the user's message.\n"
    "- Never invent facts about the CV, job, or score that are not in the "
    "context provided. If you don't know, say so.\n"
    "- When genuinely unsure whether the user wants to act, answer only and "
    "leave the gate pending - the buttons are always there as a fallback.\n\n"
    "Honesty about fit and improvements - this is the part users complain "
    "loudest about when it's wrong:\n"
    "- Never say the CV is 'well tailored', 'looks great', or similar unless "
    "the session context actually supports it (high/excellent compatibility "
    "AND at least one bullet was actually rewritten). A medium/low score, "
    "missing_skills, or zero selected/rewritten bullets means there IS room "
    "to improve - say so plainly instead of reassuring.\n"
    "- When asked what to improve, or whether there's anything to improve, "
    "read `missing_skills` and `transferable_skills` from the context and "
    "answer with the SPECIFIC skill names, not a generic 'looks good'. If a "
    "missing skill has no transferable bridge in the CV, say the pipeline "
    "cannot invent that experience, and suggest the user add it themselves "
    "if they genuinely have it (a course, a side project, brief exposure) - "
    "that unlocks real rewriting on the next pass.\n"
    "- If `bullets_selected` or `bullets_rewritten` is 0, say plainly that "
    "nothing was actually changed in the CV at that step, and why (usually: "
    "nothing in the CV was relevant enough), rather than implying the CV was "
    "already good."
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
        transferable = gap.get("transferable_skills") or []
        matched = _skill_names(gap.get("matched_skills") or [])
        if matched:
            parts.append(f"matched_skills: {matched}")
        if transferable:
            parts.append(
                "transferable_skills (real gaps the CV can partially bridge): "
                + json.dumps(
                    [
                        {"required_skill": t.get("required_skill"), "bridge_bullet": t.get("bridge_bullet")}
                        for t in transferable
                        if isinstance(t, dict)
                    ],
                    default=str,
                )[:1500]
            )
        if missing:
            parts.append(
                f"missing_skills (true gaps - the CV has NOTHING to bridge these, cannot be "
                f"invented by rewriting): {missing}"
            )

    selected = state.get("selected_content") or {}
    rewritten = state.get("rewritten_content") or {}
    bullets_selected = len(selected.get("selected_bullets") or [])
    bullets_rewritten = len(rewritten.get("rewritten_bullets") or [])
    parts.append(f"bullets_selected: {bullets_selected}")
    parts.append(f"bullets_rewritten: {bullets_rewritten}")
    if rewritten.get("updated_summary"):
        parts.append(f"updated_summary: {rewritten['updated_summary']}")

    files = state.get("generated_files") or []
    if files:
        parts.append("Generated files: " + ", ".join(f.get("filename", "") for f in files))
    return "\n".join(parts)


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
                "User message:\n{text}"
            )),
        ])
        chain = prompt | structured_llm
        decision: ChatDecision = chain.invoke({
            "gate": _gate_context_block(gate),
            "state": _state_context_block(state),
            "text": text,
        })
    except Exception as exc:  # noqa: BLE001 - defensive fallback for any LLM/parse failure
        return ChatTurn(
            reply=(
                "Sorry, I couldn't process that message just now "
                f"({exc}). You can use the buttons below, or try again."
            ),
            action=None,
        )

    if decision.action is None:
        return ChatTurn(reply=decision.reply, action=None)

    # Validate the proposed action against the actual gate before trusting it.
    if gate is None or decision.action.action not in gate.allowed_actions:
        return ChatTurn(reply=decision.reply, action=None)
    if decision.action.action == "choose":
        if not gate.choices or decision.action.choice not in gate.choices:
            return ChatTurn(reply=decision.reply, action=None)
    if decision.action.action == "edit" and not (decision.action.feedback or "").strip():
        return ChatTurn(reply=decision.reply, action=None)

    try:
        resolution = GateResolution(
            action=decision.action.action,
            feedback=decision.action.feedback,
            choice=decision.action.choice,
        )
    except Exception:
        return ChatTurn(reply=decision.reply, action=None)

    return ChatTurn(reply=decision.reply, action=resolution)
