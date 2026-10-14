"""
Selection review: match labels instead of raw scores, and leaving bullets out.

- each selected bullet gets a match word (strong >= 70%, partial >= 35%, weak
  below), the posting requirements it shows (from the rubric), and a leave-out
  suggestion only when it is a weak match that shows no requirement;
- the three choices: tailor it (default), keep as is (not rewritten), leave
  out (removed from the tailored CV; the user's CV keeps it).

Run with::

    cd backend
    pytest tests/test_leave_out.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langgraph.types import Command

from app.agents.orchestrator.gates import BulletDecision, GateResolution
from app.agents.orchestrator.nodes_job_tailoring import (
    _finalize_selection_decisions,
    _index_cv_bullets,
    _with_match_labels,
    _without_bullets,
)

from tests.test_orchestrator_job_tailoring import (  # noqa: F401  (fixtures)
    COMPATIBILITY_REPORT,
    CV,
    JOB,
    _fake_tool,
    _interrupt_payload,
    graph,
    patched_tools,
)

RUBRIC = [
    {"requirement_id": "req.0", "text": "Python", "category": "must_have",
     "evidence_bullet_ids": ["experience.0.responsibilities.0"]},
    {"requirement_id": "ko.0", "text": "German (C1)", "category": "knockout",
     "evidence_bullet_ids": ["experience.0.responsibilities.1"]},
]


def _bullet(bullet_id: str, relevance) -> dict:
    return {"section": "experience", "original_text": "x", "relevance_score": relevance, "bullet_id": bullet_id}


@pytest.mark.parametrize("relevance, match", [(0.9, "strong"), (0.7, "strong"), (0.5, "partial"),
                                              (0.35, "partial"), (0.2, "weak"), ("n/a", None)])
def test_match_words_by_relevance(relevance, match):
    assert _with_match_labels(_bullet("projects.0.outcomes.0", relevance), RUBRIC)["match"] == match


def test_leave_out_is_suggested_only_for_weak_bullets_that_show_nothing():
    weak_unrelated = _with_match_labels(_bullet("projects.0.outcomes.0", 0.2), RUBRIC)
    assert weak_unrelated["leave_out_recommended"] is True and weak_unrelated["shows"] == []
    # Weak by the model's estimate, but the rubric says it proves a must-have.
    weak_but_evidence = _with_match_labels(_bullet("experience.0.responsibilities.0", 0.2), RUBRIC)
    assert weak_but_evidence["leave_out_recommended"] is False and weak_but_evidence["shows"] == ["Python"]
    # Knockouts are not skill evidence.
    knockout_only = _with_match_labels(_bullet("experience.0.responsibilities.1", 0.2), RUBRIC)
    assert knockout_only["shows"] == [] and knockout_only["leave_out_recommended"] is True
    assert _with_match_labels(_bullet("projects.0.outcomes.0", 0.5), RUBRIC)["leave_out_recommended"] is False


def test_decisions_split_into_tailor_keep_and_leave_out():
    bullets = [_bullet("a", 0.9), _bullet("b", 0.5), _bullet("c", 0.1), _bullet(None, 0.1)]
    decisions = {"b": BulletDecision(choice="exclude"), "c": BulletDecision(choice="leave_out")}
    kept, left_out = _finalize_selection_decisions(bullets, decisions)
    assert [b["bullet_id"] for b in kept] == ["a", None]
    assert left_out == ["c"]


def test_leave_out_is_a_valid_decision():
    assert GateResolution(action="approve", decisions={"x": {"choice": "leave_out"}}).decisions["x"].choice == "leave_out"


def test_left_out_bullets_are_removed_with_the_remaining_order_kept():
    cv = {**CV, "experience": [{**CV["experience"][0], "responsibilities": ["one", "two", "three"]}],
          "projects": [{"name": "P", "description": "desc", "outcomes": ["o1"]}]}
    index = _index_cv_bullets(cv)
    out = _without_bullets(cv, ["experience.0.responsibilities.0", "experience.0.responsibilities.2",
                                "projects.0.description"], index)
    assert out["experience"][0]["responsibilities"] == ["two"]
    assert out["projects"][0]["description"] == ""
    assert cv["experience"][0]["responsibilities"] == ["one", "two", "three"]  # the input is untouched


SELECTED = {
    "selected_bullets": [
        {"section": "experience", "original_text": "Designed the first algorithm", "relevance_score": 0.9,
         "reason": "algorithms", "bullet_id": "experience.0.responsibilities.0"},
        {"section": "experience", "original_text": "Wrote technical notes", "relevance_score": 0.1,
         "reason": "unrelated", "bullet_id": "experience.0.responsibilities.1"},
    ],
    "section_order": [], "sections_to_emphasize": [], "items_to_de_emphasize": [],
}


def test_a_left_out_bullet_is_not_rewritten_and_not_in_the_tailored_cv(graph, patched_tools):  # noqa: F811
    rewrites: list = []

    def rewrite(kwargs):
        rewrites.append(json.loads(kwargs["selected_content_json"]))
        return json.dumps({"rewritten_bullets": [], "updated_summary": "", "keywords_inserted": []})

    with patch("app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
               _fake_tool(lambda _: json.dumps(SELECTED))), \
         patch("app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content", _fake_tool(rewrite)):
        config = {"configurable": {"thread_id": "leave-out"}}
        graph.invoke({"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB}, config=config)
        payload = _interrupt_payload(graph.invoke(Command(resume={"action": "approve"}), config=config))
        assert payload["step"] == "approve_selection"
        labels = {b["bullet_id"]: (b["match"], b["leave_out_recommended"]) for b in payload["preview"]["selected_bullets"]}
        assert labels == {"experience.0.responsibilities.0": ("strong", False),
                          "experience.0.responsibilities.1": ("weak", True)}

        graph.invoke(Command(resume={"action": "approve", "decisions": {
            "experience.0.responsibilities.1": {"choice": "leave_out"}}}), config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # the rewrites

    assert [b["bullet_id"] for b in rewrites[0]["selected_bullets"]] == ["experience.0.responsibilities.0"]
    values = graph.get_state(config).values
    assert values["tailored_cv_data"]["experience"][0]["responsibilities"] == ["Designed the first algorithm"]
    assert values["cv_data"]["experience"][0]["responsibilities"] == CV["experience"][0]["responsibilities"]
    assert any("1 left out of the tailored CV" in m.content for m in values["messages"])
