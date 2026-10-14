"""
Strategist: the holistic tailoring plan, checked against the CV and reviewed
together with the score.

- validation decides what's trusted: a keyword is allowed only when a fact on
  the CV names it, featured bullets / sections / projects must exist, and the
  headline goes through the fact check;
- the plan runs before the score review, edit redrafts it with the user's
  notes, and a failed draft falls back to the rule-based strategy;
- selection and rewriting see the parts of the plan they act on.

Run with::

    cd backend
    pytest tests/test_strategist.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from app.agents.strategist_agent import plan_tailoring_strategy, validate_plan
from app.agents.writer_agent import _format_plan
from app.models.schemas import KeywordPlanItem, StrategistPlan
from app.services.bullet_index import index_cv_bullets
from app.services.fact_ledger import build_fact_ledger

from tests.test_orchestrator_job_tailoring import (  # noqa: F401  (fixtures)
    CV,
    JOB,
    PLAN,
    _fake_tool,
    _interrupt_payload,
    graph,
    patched_tools,
)

SCV = {
    "skills": ["Python", "PostgreSQL"],
    "experience": [{"position": "Engineer", "company": "X", "duration": "2021 - 2023",
                    "responsibilities": ["Built APIs in Python", "Tuned PostgreSQL queries"]}],
    "projects": [{"name": "A", "technologies": [], "outcomes": []}, {"name": "B", "technologies": [], "outcomes": []}],
}
SINDEX = index_cv_bullets(SCV)
SLEDGER = build_fact_ledger(SCV, SINDEX)
RUBRIC = [{"requirement_id": "req.0", "text": "Python"}, {"requirement_id": "req.1", "text": "SQL"}]


def _plan(**overrides) -> StrategistPlan:
    base = dict(positioning="Backend engineer.", headline=None, cv_language="english", convention="international")
    return StrategistPlan(**{**base, **overrides})


def _validate(plan: StrategistPlan):
    return validate_plan(plan, rubric=RUBRIC, ledger=SLEDGER, bullet_index=SINDEX, project_count=2)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_keywords_are_allowed_only_when_the_cv_names_them():
    plan, notes = _validate(_plan(keyword_plan=[
        KeywordPlanItem(keyword="Python", requirement_id="req.0", evidence_fact_ids=["skill.python"]),
        KeywordPlanItem(keyword="PostgreSQL", requirement_id="req.1", evidence_fact_ids=["skill.python"]),
        KeywordPlanItem(keyword="Kubernetes", requirement_id="req.9", allowed=True, evidence_fact_ids=["skill.python"]),
    ]))
    by_keyword = {k.keyword: k for k in plan.keyword_plan}
    assert by_keyword["Python"].allowed and by_keyword["Python"].evidence_fact_ids == ["skill.python"]
    # Wrong citation, but the CV does show it: allowed with the real evidence.
    assert by_keyword["PostgreSQL"].allowed and "skill.postgresql" in by_keyword["PostgreSQL"].evidence_fact_ids
    # The model's allowed=True is not trusted.
    kubernetes = by_keyword["Kubernetes"]
    assert not kubernetes.allowed and kubernetes.evidence_fact_ids == []
    assert kubernetes.requirement_id == ""  # req.9 isn't in the rubric
    assert any("Kubernetes" in n for n in notes)


def test_featured_bullets_sections_and_projects_must_exist():
    plan, notes = _validate(_plan(
        bullets_to_feature=["experience.0.responsibilities.1", "experience.7.responsibilities.0"],
        section_order=["work", "Projects", "hobbies", "skills", "work"],
        project_order=[1, 5, 0, 1],
    ))
    assert plan.bullets_to_feature == ["experience.0.responsibilities.1"]
    assert plan.section_order == ["experience", "projects", "skills"]
    assert plan.project_order == [1, 0]
    assert any("featured bullets" in n for n in notes)


def test_an_unsupported_headline_is_dropped():
    kept, _ = _validate(_plan(headline="Backend engineer working with Python and PostgreSQL"))
    assert kept.headline == "Backend engineer working with Python and PostgreSQL"
    dropped, notes = _validate(_plan(headline="Cloud engineer with Kubernetes and Python"))
    assert dropped.headline is None
    assert any("headline" in n and "kubernetes" in n for n in notes)


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #

class _StubModel:
    def __init__(self, result):
        self.result = result
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        def answer(prompt_value):
            self.prompts.append(prompt_value.to_string())
            if isinstance(self.result, Exception):
                raise self.result
            return self.result
        return RunnableLambda(answer)


def test_the_tool_wraps_the_posting_and_passes_the_users_notes():
    model = _StubModel(_plan())
    with patch("app.agents.strategist_agent.chat_model", return_value=model):
        out = json.loads(plan_tailoring_strategy.invoke({
            "cv_json": "{}", "job_json": '{"job_title": "Dev"}', "report_json": "{}",
            "feedback": "- Lead with projects",
        }))
    assert out["positioning"] == "Backend engineer."
    assert "<job_posting>" in model.prompts[0] and "Lead with projects" in model.prompts[0]


def test_the_tool_reports_failures_as_errors():
    with patch("app.agents.strategist_agent.chat_model", return_value=_StubModel(RuntimeError("quota"))):
        out = json.loads(plan_tailoring_strategy.invoke({"cv_json": "{}", "job_json": "{}", "report_json": "{}"}))
    assert out == {"error": "quota"}


# --------------------------------------------------------------------------- #
# What the writer sees
# --------------------------------------------------------------------------- #

def test_selection_and_rewriting_see_their_parts_of_the_plan():
    plan, _ = _validate(_plan(
        section_order=["experience", "skills"], bullets_to_feature=["experience.0.responsibilities.0"],
        headline="Engineer working with Python",
        keyword_plan=[KeywordPlanItem(keyword="Python", evidence_fact_ids=["skill.python"]),
                      KeywordPlanItem(keyword="Kubernetes")],
    ))
    strategy = {"plan": plan.model_dump()}
    select = _format_plan(strategy, "select")
    assert "experience, skills" in select and "experience.0.responsibilities.0" in select
    assert "Kubernetes" not in select
    rewrite = _format_plan(strategy, "rewrite")
    assert "Engineer working with Python" in rewrite
    assert "supports (use where they fit naturally): Python" in rewrite
    assert "does NOT support (never use): Kubernetes" in rewrite
    assert _format_plan({}, "select") == "None."


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

def _start(graph, thread):  # noqa: F811
    config = {"configurable": {"thread_id": thread}}
    return config, graph.invoke({"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB}, config=config)


def test_the_plan_is_reviewed_with_the_score(graph, patched_tools):  # noqa: F811
    _, result = _start(graph, "plan-review")
    payload = _interrupt_payload(result)
    assert payload["step"] == "present_score"
    assert payload["allowed_actions"] == ["approve", "edit", "reject"]
    plan = payload["preview"]["plan"]
    assert plan["keywords_to_use"] == ["Python"]
    assert plan["keywords_not_on_cv"] == ["Kubernetes"]
    assert plan["featured_bullets"] == [{"bullet_id": "experience.0.responsibilities.0",
                                         "text": "Designed the first algorithm"}]
    assert any("Kubernetes" in n for n in plan["notes"])
    assert payload["preview"]["aggregate_score"] is not None  # the fit is still there


def test_edit_redrafts_the_plan_with_the_users_notes(graph, patched_tools):  # noqa: F811
    calls: list[dict] = []

    def draft(kwargs):
        calls.append(kwargs)
        return json.dumps(PLAN)

    with patch("app.agents.orchestrator.nodes_job_tailoring.plan_tailoring_strategy", _fake_tool(draft)):
        config, _ = _start(graph, "plan-edit")
        result = graph.invoke(Command(resume={"action": "edit", "feedback": "Lead with projects"}), config=config)
    assert _interrupt_payload(result)["step"] == "present_score"
    assert len(calls) == 2
    assert calls[1]["feedback"] == "- Lead with projects"


def test_approving_the_plan_hands_it_to_selection(graph, patched_tools):  # noqa: F811
    seen: dict = {}

    def select(kwargs):
        seen["strategy"] = json.loads(kwargs["tailoring_strategy_json"])
        return json.dumps({"selected_bullets": [], "section_order": []})

    with patch("app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content", _fake_tool(select)):
        config, _ = _start(graph, "plan-approve")
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_selection"
    assert seen["strategy"]["plan"]["positioning"] == PLAN["positioning"]
    assert seen["strategy"]["directives"]  # the rule-based directives are still there


def test_a_failed_draft_falls_back_to_the_rule_based_strategy(graph, patched_tools):  # noqa: F811
    with patch("app.agents.orchestrator.nodes_job_tailoring.plan_tailoring_strategy",
               _fake_tool(lambda _: json.dumps({"error": "provider timeout"}))):
        config, result = _start(graph, "plan-failed")
    payload = _interrupt_payload(result)
    assert payload["step"] == "present_score"
    assert payload["preview"]["plan"] is None
    assert "rule-based strategy" in payload["narration"]
    assert any("provider timeout" in m.content for m in result["messages"])
    assert _interrupt_payload(graph.invoke(Command(resume={"action": "approve"}), config=config))["step"] == \
        "approve_selection"


@pytest.mark.parametrize("bad", ['{"positioning": 3}', "not json"])
def test_a_malformed_draft_is_reported_not_raised(graph, patched_tools, bad):  # noqa: F811
    with patch("app.agents.orchestrator.nodes_job_tailoring.plan_tailoring_strategy", _fake_tool(lambda _: bad)):
        _, result = _start(graph, f"plan-bad-{len(bad)}")
    assert _interrupt_payload(result)["preview"]["plan"] is None
