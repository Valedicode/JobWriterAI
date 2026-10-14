"""
Structured human-in-the-loop gates for the orchestrator graph.

Nodes that need user input emit a GatePayload via LangGraph's interrupt()
function. The API layer surfaces the payload to the frontend, the frontend
renders an approval card / choice card, and the user's reply travels back as a
GateResolution. The graph is resumed with Command(resume=GateResolution.dict()).

Gate kinds:
- "approval" : Approve / Reject / (optionally) Edit a structured preview.
- "choice"   : Pick one option from a fixed list (e.g. PDF / DOCX / both).
- "input"    : Collect a single free-text value (submitted via action='edit').

The two enums (GateKind, GateAction) are exposed as plain string Literals so
they serialise cleanly through JSON without needing custom encoders.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Selection: include (tailor it), exclude (keep it as it is), leave_out (drop it
# from the tailored CV). Rewrite: accept, keep_original, custom.
BulletDecisionChoice = Literal["accept", "keep_original", "custom", "include", "exclude", "leave_out"]


class BulletDecision(BaseModel):
    """Per-bullet decision attached to a gate resolution's `decisions` map."""

    choice: BulletDecisionChoice
    text: str | None = None


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

GateKind = Literal["approval", "choice", "input"]

GateAction = Literal["approve", "reject", "edit", "choose"]


# --------------------------------------------------------------------------- #
# Payload (server -> client)
# --------------------------------------------------------------------------- #

class GatePayload(BaseModel):
    """
    Emitted by a graph node when it needs the user to make a decision.

    The dict form of this model is what gets handed to LangGraph's
    interrupt(...) call and what the API returns to the frontend.
    """

    step: str = Field(
        ...,
        description=(
            "Stable identifier for the gate, e.g. 'approve_selection', "
            "'approve_rewrite'. Used by the frontend to "
            "decide which preview component to render."
        ),
    )
    kind: GateKind = Field(
        ...,
        description="'approval' for approve/reject/edit gates, 'choice' for pick-one gates.",
    )
    narration: str = Field(
        ...,
        description="Natural-language message rendered as the latest assistant chat bubble.",
    )
    preview: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured artifact the user is reviewing (SelectedContent, "
            "RewrittenContent, CoverLetterContent, etc.). Empty dict for "
            "pure-choice gates."
        ),
    )
    allowed_actions: list[GateAction] = Field(
        ...,
        description="Actions the frontend may submit back. Must be non-empty.",
    )
    choices: list[str] | None = Field(
        default=None,
        description=(
            "For kind='choice' gates: the fixed set of valid values for "
            "GateResolution.choice. Ignored for kind='approval'."
        ),
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> "GatePayload":
        if not self.allowed_actions:
            raise ValueError("allowed_actions must contain at least one action")
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("choices must be provided for kind='choice'")
            if "choose" not in self.allowed_actions:
                raise ValueError("kind='choice' gates must allow the 'choose' action")
        if self.kind == "input" and "edit" not in self.allowed_actions:
            raise ValueError("kind='input' gates must allow the 'edit' action")
        return self


# --------------------------------------------------------------------------- #
# Resolution (client -> server)
# --------------------------------------------------------------------------- #

class GateResolution(BaseModel):
    """
    The user's reply to a GatePayload. Travels in the body of
    POST /api/orchestrator/message when kind='gate_resolution'.
    """

    action: GateAction = Field(
        ...,
        description="Which action the user picked. Must be in the GatePayload's allowed_actions.",
    )
    feedback: str | None = Field(
        default=None,
        description=(
            "Free-text edit instructions. Required when action='edit'; ignored otherwise."
        ),
    )
    choice: str | None = Field(
        default=None,
        description=(
            "Selected option for kind='choice' gates. Required when action='choose'; "
            "must be one of the GatePayload.choices values."
        ),
    )
    decisions: dict[str, BulletDecision] | None = Field(
        default=None,
        description=(
            "Per-bullet decisions keyed by bullet_id, for approve_selection "
            "and approve_rewrite gates. Bullets with no entry default to "
            "'accept'/'include'."
        ),
    )
    answers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Answers keyed by question id, for gates whose preview lists "
            "`questions` (confirm_eligibility). Unanswered ids are skipped."
        ),
    )

    @model_validator(mode="after")
    def _validate_shape(self) -> "GateResolution":
        if self.action == "edit" and not (self.feedback and self.feedback.strip()):
            raise ValueError("feedback is required when action='edit'")
        if self.action == "choose" and not (self.choice and self.choice.strip()):
            raise ValueError("choice is required when action='choose'")
        return self
