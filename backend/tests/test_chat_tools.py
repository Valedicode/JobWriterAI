"""
Tool-calling chat built on the gates.

Drives /api/orchestrator/message with the chat model's choice stubbed, and
checks each tool's routing:

- answering leaves the gate as it was and is kept as short-term memory;
- edit_bullet re-runs one rewrite with every other bullet locked, and only
  at approve_rewrite;
- rerun_from and update_profile become a proposal at the confirm_change gate;
  nothing changes until it is approved, approving runs forward from the
  target step through every gate again, and cancelling returns to the gate
  the user was on.

After every chat turn the session is paused on a gate (or finished), never
past one.

Run with::

    cd backend
    pytest tests/test_chat_tools.py -v
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import app.api.orchestrator as orchestrator_module
from app.agents.orchestrator.chat_actions import CHAT_MEMORY_MESSAGES, ChatActionRefused, propose_rerun
from app.agents.orchestrator.chat_agent import (
    ChatDecision,
    ChatTurn,
    EditBullet,
    ResolveGate,
    RerunFrom,
    UpdateProfile,
    interpret_chat,
)
from app.agents.orchestrator.gates import GatePayload, GateResolution
from app.agents.orchestrator.nodes_job_tailoring import _index_cv_bullets

from tests.test_orchestrator_api import enabled_client  # noqa: F401  (fixture)
from tests.test_orchestrator_job_tailoring import (
    COMPATIBILITY_REPORT,
    COVER_LETTER_CONTENT,
    CV,
    JOB,
    JOB_SUMMARY,
    STRATEGY,
)

BULLET_IDS = list(_index_cv_bullets(CV))  # two experience bullets


def _rewrite_of(bullet_id: str, text: str) -> dict[str, Any]:
    original = _index_cv_bullets(CV)[bullet_id]["text"]
    return {"original": original, "rewritten": text, "confidence": 0.9,
            "keywords_added": [], "bullet_id": bullet_id}


SELECTED = {
    "selected_bullets": [
        {"section": "experience", "original_text": _index_cv_bullets(CV)[b]["text"],
         "relevance_score": 0.9, "reason": "relevant", "bullet_id": b}
        for b in BULLET_IDS
    ],
    "section_order": ["experience", "skills", "education"],
    "sections_to_emphasize": ["experience"],
    "items_to_de_emphasize": [],
}


class _Tools:
    """Recording stand-ins for the step tools."""

    def __init__(self):
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.setdefault(name, []).append(kwargs)

    def count(self, name: str) -> int:
        return len(self.calls.get(name, []))

    def stubs(self) -> dict[str, Any]:
        def tool(name, result):
            def invoke(kwargs):
                self._record(name, kwargs)
                return result(kwargs) if callable(result) else json.dumps(result)
            return SimpleNamespace(invoke=invoke)

        def rewrite(kwargs):
            locked = {b["bullet_id"]: b for b in json.loads(kwargs["locked_bullets_json"])}
            n = self.count("rewrite_enhance_content")
            bullets = [locked.get(b) or _rewrite_of(b, f"Rewrite {n} of {b}") for b in BULLET_IDS]
            return json.dumps({"rewritten_bullets": bullets, "updated_summary": "", "keywords_inserted": []})

        return {
            "generate_job_summary": tool("generate_job_summary", JOB_SUMMARY),
            "calculate_compatibility_score_v2": tool("calculate_compatibility_score_v2", COMPATIBILITY_REPORT),
            "decide_tailoring_strategy": tool("decide_tailoring_strategy", STRATEGY),
            "select_prioritize_content": tool("select_prioritize_content", SELECTED),
            "rewrite_enhance_content": tool("rewrite_enhance_content", rewrite),
            "generate_cv_pdf": tool(
                "generate_cv_pdf",
                lambda kw: f"CV PDF generated successfully! The file '{kw['output_filename']}' is ready for download.",
            ),
            "generate_cover_letter_content": tool("generate_cover_letter_content", COVER_LETTER_CONTENT),
            "generate_cover_letter_pdf": tool(
                "generate_cover_letter_pdf",
                lambda kw: f"Cover letter PDF generated successfully! The file '{kw['output_filename']}' is ready.",
            ),
        }


@pytest.fixture
def tools():
    recorder = _Tools()
    patches = [
        patch(f"app.agents.orchestrator.nodes_job_tailoring.{name}", stub)
        for name, stub in recorder.stubs().items()
    ]
    for p in patches:
        p.start()
    yield recorder
    for p in patches:
        p.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _start(client) -> str:
    body = client.post(
        "/api/orchestrator/start", json={"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}
    ).json()
    assert body["pending_gate"]["step"] == "present_score"
    return body["session_id"]


def _approve(client, session_id: str, times: int = 1) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for _ in range(times):
        body = client.post("/api/orchestrator/message", json={
            "session_id": session_id, "kind": "gate_resolution", "resolution": {"action": "approve"},
        }).json()
    return body


def _resolve(client, session_id: str, action: str) -> dict[str, Any]:
    return client.post("/api/orchestrator/message", json={
        "session_id": session_id, "kind": "gate_resolution", "resolution": {"action": action},
    }).json()


def _chat(client, session_id: str, turn: ChatTurn, text: str = "chat message", seen: list | None = None):
    def interpret(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return turn

    with patch.object(orchestrator_module, "interpret_chat", side_effect=interpret):
        resp = client.post("/api/orchestrator/message",
                           json={"session_id": session_id, "kind": "chat", "text": text})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _state(session_id: str) -> dict[str, Any]:
    return orchestrator_module._get_graph().get_state({"configurable": {"thread_id": session_id}}).values


def _pending_step(client, session_id: str) -> str | None:
    gate = client.get(f"/api/orchestrator/state/{session_id}").json()["pending_gate"]
    return gate["step"] if gate else None


# --------------------------------------------------------------------------- #
# Answering and memory
# --------------------------------------------------------------------------- #

def test_answer_keeps_the_gate_and_is_remembered(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    seen: list = []
    body = _chat(enabled_client, sid, ChatTurn(reply="It measures keyword overlap."),
                 text="what is ATS density?", seen=seen)
    assert body["pending_gate"]["step"] == "present_score"
    assert _pending_step(enabled_client, sid) == "present_score"
    assert _state(sid)["chat_history"] == [
        {"role": "user", "content": "what is ATS density?"},
        {"role": "assistant", "content": "It measures keyword overlap."},
    ]

    _chat(enabled_client, sid, ChatTurn(reply="Yes."), text="and is that good?", seen=seen)
    assert len(seen[1]["state"]["chat_history"]) == 2  # the model saw the earlier turn
    # The remembered gate still takes a normal answer.
    assert _approve(enabled_client, sid)["pending_gate"]["step"] == "approve_selection"


def test_chat_memory_is_bounded(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    for i in range(CHAT_MEMORY_MESSAGES):
        _chat(enabled_client, sid, ChatTurn(reply=f"answer {i}"), text=f"question {i}")
    history = _state(sid)["chat_history"]
    assert len(history) == CHAT_MEMORY_MESSAGES
    assert history[-1]["content"] == f"answer {CHAT_MEMORY_MESSAGES - 1}"
    assert _pending_step(enabled_client, sid) == "present_score"


def test_memory_after_the_flow_ended_leaves_it_ended(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    assert _resolve(enabled_client, sid, "reject")["done"] is True
    body = _chat(enabled_client, sid, ChatTurn(reply="You stopped at the score."))
    assert body["done"] is True
    assert _pending_step(enabled_client, sid) is None
    assert len(_state(sid)["chat_history"]) == 2


# --------------------------------------------------------------------------- #
# edit_bullet
# --------------------------------------------------------------------------- #

def test_edit_bullet_reruns_only_that_bullet(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    gate = _approve(enabled_client, sid, times=2)["pending_gate"]
    assert gate["step"] == "approve_rewrite"
    target, other = BULLET_IDS
    kept_text = next(b["rewritten"] for b in gate["preview"]["rewritten_bullets"] if b["bullet_id"] == other)

    body = _chat(enabled_client, sid, ChatTurn(
        reply="Changing that bullet.",
        call=EditBullet(tool="edit_bullet", bullet_id=target, instruction="mention the team size"),
    ))

    assert body["pending_gate"]["step"] == "approve_rewrite"  # back at the same review
    locked = json.loads(tools.calls["rewrite_enhance_content"][-1]["locked_bullets_json"])
    assert [b["bullet_id"] for b in locked] == [other]
    bullets = {b["bullet_id"]: b["rewritten"] for b in body["pending_gate"]["preview"]["rewritten_bullets"]}
    assert bullets[other] == kept_text
    assert bullets[target] == f"Rewrite 2 of {target}"  # regenerated in the second pass
    feedback = _state(sid)["tailoring_strategy"]["user_feedback"][-1]["text"]
    assert target in feedback and "team size" in feedback


def test_edit_bullet_outside_the_rewrite_review_is_refused(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    body = _chat(enabled_client, sid, ChatTurn(
        reply="Sure, changing it.",
        call=EditBullet(tool="edit_bullet", bullet_id=BULLET_IDS[0], instruction="shorter"),
    ))
    assert "while you're reviewing the rewrites" in body["narrations"][0]
    assert body["pending_gate"]["step"] == "present_score"
    assert tools.count("rewrite_enhance_content") == 0


def test_edit_bullet_with_an_unknown_id_is_refused(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid, times=2)
    body = _chat(enabled_client, sid, ChatTurn(
        reply="Changing it.", call=EditBullet(tool="edit_bullet", bullet_id="experience.9.x.0", instruction="x"),
    ))
    assert "couldn't find that bullet" in body["narrations"][0]
    assert tools.count("rewrite_enhance_content") == 1


# --------------------------------------------------------------------------- #
# rerun_from
# --------------------------------------------------------------------------- #

def _rerun(step: str, note: str | None = None) -> ChatTurn:
    return ChatTurn(reply="Let's redo that.", call=RerunFrom(tool="rerun_from", step=step, note=note))


def test_rerun_is_only_a_proposal_until_confirmed(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid, times=2)  # at approve_rewrite
    before = copy.deepcopy(_state(sid))

    body = _chat(enabled_client, sid, _rerun("select"))
    assert body["pending_gate"]["step"] == "confirm_change"
    assert "bullet selection" in body["pending_gate"]["preview"]["change"]
    after = _state(sid)
    for key in ("rewritten_content", "selected_content", "cv_data", "tailoring_strategy"):
        assert after[key] == before[key], key
    assert tools.count("select_prioritize_content") == 1


def test_confirmed_rerun_runs_forward_through_the_gates(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid, times=2)
    _chat(enabled_client, sid, _rerun("select", note="focus on algorithms"))

    body = _approve(enabled_client, sid)  # confirm the change
    assert body["pending_gate"]["step"] == "approve_selection"
    assert tools.count("select_prioritize_content") == 2
    assert _state(sid)["rewritten_content"] is None  # the old rewrites are gone
    strategy = json.loads(tools.calls["select_prioritize_content"][-1]["tailoring_strategy_json"])
    assert strategy["user_feedback"][-1] == {"stage": "chat", "text": "focus on algorithms"}
    # And on through the next gate as normal.
    assert _approve(enabled_client, sid)["pending_gate"]["step"] == "approve_rewrite"


def test_cancelled_rerun_returns_to_the_same_gate(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    gate = _approve(enabled_client, sid, times=2)["pending_gate"]
    _chat(enabled_client, sid, _rerun("score"))

    body = _resolve(enabled_client, sid, "reject")
    assert body["pending_gate"] == gate
    assert tools.count("calculate_compatibility_score_v2") == 1
    assert _state(sid)["chat_proposal"] is None


def test_rerun_of_a_step_that_has_not_run_is_refused(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    body = _chat(enabled_client, sid, _rerun("rewrite"))
    assert "hasn't run yet" in body["narrations"][0]
    assert body["pending_gate"]["step"] == "present_score"


def test_rerun_after_tailoring_starts_from_the_untailored_cv(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    body = _approve(enabled_client, sid, times=3)  # rewrites approved, CV assembled and rendered
    assert body["pending_gate"]["step"] == "cover_letter_language"
    assert body["generated_files"]
    tailored = _state(sid)["tailored_cv_data"]["experience"][0]["responsibilities"]
    assert tailored != CV["experience"][0]["responsibilities"]
    # Step 6 no longer overwrites the user's own CV.
    assert _state(sid)["cv_data"]["experience"][0]["responsibilities"] == CV["experience"][0]["responsibilities"]

    proposal = _chat(enabled_client, sid, _rerun("rewrite"))["pending_gate"]
    assert any("leave this chat" in line for line in proposal["preview"]["what_happens"])
    body = _approve(enabled_client, sid)
    assert body["pending_gate"]["step"] == "approve_rewrite"
    assert body["generated_files"] == []
    state = _state(sid)
    assert state["tailored_cv_data"] is None
    assert state["cv_data"]["experience"][0]["responsibilities"] == CV["experience"][0]["responsibilities"]
    cv_sent = json.loads(tools.calls["rewrite_enhance_content"][-1]["cv_json"])
    assert cv_sent["experience"][0]["responsibilities"] == CV["experience"][0]["responsibilities"]


def test_cover_letter_rerun_keeps_the_tailored_cv(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid, times=3)
    body = enabled_client.post("/api/orchestrator/message", json={
        "session_id": sid, "kind": "gate_resolution", "resolution": {"action": "choose", "choice": "english"},
    }).json()
    assert body["pending_gate"]["step"] == "approve_cover_letter"
    tailored = copy.deepcopy(_state(sid)["tailored_cv_data"])
    assert tailored

    _chat(enabled_client, sid, _rerun("cover_letter"))
    body = _approve(enabled_client, sid)
    assert body["pending_gate"]["step"] == "cover_letter_language"
    assert _state(sid)["tailored_cv_data"] == tailored
    assert _state(sid)["cover_letter_content"] is None


def test_rerun_after_the_flow_ended(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    assert _resolve(enabled_client, sid, "reject")["done"] is True
    assert _chat(enabled_client, sid, _rerun("score"))["pending_gate"]["step"] == "confirm_change"
    body = _approve(enabled_client, sid)
    assert body["pending_gate"]["step"] == "present_score"
    assert tools.count("calculate_compatibility_score_v2") == 2


def test_a_second_proposal_waits_for_the_first(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid, times=2)
    _chat(enabled_client, sid, _rerun("select"))
    body = _chat(enabled_client, sid, _rerun("score"))
    assert "already a change waiting" in body["narrations"][0]
    assert body["pending_gate"]["step"] == "confirm_change"
    assert "bullet selection" in body["pending_gate"]["preview"]["change"]


# --------------------------------------------------------------------------- #
# update_profile
# --------------------------------------------------------------------------- #

def _add_skill(value: str) -> ChatTurn:
    return ChatTurn(reply="I can add that.", call=UpdateProfile(tool="update_profile", kind="skill", value=value))


def test_profile_update_is_confirmed_then_rescored(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _approve(enabled_client, sid)  # at approve_selection
    body = _chat(enabled_client, sid, _add_skill("Docker"), text="I used Docker in a university course")
    gate = body["pending_gate"]
    assert gate["step"] == "confirm_change"
    assert "Docker" in gate["preview"]["change"]
    assert "university course" in " ".join(gate["preview"]["what_happens"])
    assert "Docker" not in _state(sid)["cv_data"]["skills"]

    body = _approve(enabled_client, sid)
    assert body["pending_gate"]["step"] == "present_score"
    state = _state(sid)
    assert "Docker" in state["cv_data"]["skills"]
    fact = state["fact_ledger"]["skill.docker"]
    assert fact["source"] == "user_added" and fact["user_verified"] is True
    assert fact["source_text"] == "I used Docker in a university course"
    assert fact["entities"] == ["docker"]
    scored_cv = json.loads(tools.calls["calculate_compatibility_score_v2"][-1]["cv_json"])
    assert "Docker" in scored_cv["skills"]


def test_profile_language_update(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _chat(enabled_client, sid, ChatTurn(
        reply="Adding German.",
        call=UpdateProfile(tool="update_profile", kind="language", value="German", fluency="C1"),
    ), text="I speak German at C1")
    _approve(enabled_client, sid)
    assert _state(sid)["cv_data"]["languages"] == [{"language": "German", "fluency": "C1"}]


def test_a_skill_already_on_the_cv_is_not_proposed(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    body = _chat(enabled_client, sid, _add_skill("python"))
    assert "already in your CV's skills" in body["narrations"][0]
    assert body["pending_gate"]["step"] == "present_score"


def test_cancelled_profile_update_changes_nothing(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _chat(enabled_client, sid, _add_skill("Docker"))
    body = _resolve(enabled_client, sid, "reject")
    assert body["pending_gate"]["step"] == "present_score"
    assert "Docker" not in _state(sid)["cv_data"]["skills"]
    assert "skill.docker" not in _state(sid)["fact_ledger"]


def test_profile_updates_survive_a_later_rerun(enabled_client, tools):  # noqa: F811
    sid = _start(enabled_client)
    _chat(enabled_client, sid, _add_skill("Docker"))
    _approve(enabled_client, sid)  # confirm -> rescored, at present_score
    _approve(enabled_client, sid, times=2)  # to approve_rewrite
    _chat(enabled_client, sid, _rerun("score"))
    _approve(enabled_client, sid)
    assert "Docker" in _state(sid)["cv_data"]["skills"]
    assert _state(sid)["fact_ledger"]["skill.docker"]["source"] == "user_added"


# --------------------------------------------------------------------------- #
# Chat agent: tool choice -> ChatTurn
# --------------------------------------------------------------------------- #

class _StubChatModel:
    def __init__(self, decision: ChatDecision):
        self.decision = decision

    def __call__(self, role, *, temperature=None):
        return self

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda _: self.decision)


_GATE = GatePayload(step="present_score", kind="approval", narration="", allowed_actions=["approve", "reject"])


def _interpret(decision: ChatDecision) -> ChatTurn:
    with patch("app.agents.orchestrator.chat_agent.chat_model", _StubChatModel(decision)):
        return interpret_chat(text="hi", gate=_GATE, state={})


def test_gate_answers_outside_the_allowed_actions_become_plain_replies():
    turn = _interpret(ChatDecision(reply="Editing.", call=ResolveGate(tool="resolve_gate", action="edit", feedback="x")))
    assert turn.action is None and turn.call is None
    turn = _interpret(ChatDecision(reply="Approving.", call=ResolveGate(tool="resolve_gate", action="approve")))
    assert turn.action == GateResolution(action="approve")


def test_other_tools_pass_through_to_the_api():
    call = RerunFrom(tool="rerun_from", step="score")
    assert _interpret(ChatDecision(reply="Sure.", call=call)).call == call


def test_a_turn_cannot_carry_both_kinds_of_action():
    with pytest.raises(ValueError):
        ChatTurn(reply="x", action=GateResolution(action="approve"), call=RerunFrom(tool="rerun_from", step="score"))


def test_reruns_are_job_tailoring_only():
    with pytest.raises(ChatActionRefused):
        propose_rerun(None, {}, {"flow": "cv_review"}, pending_node=None, pending_step=None, step="score")
