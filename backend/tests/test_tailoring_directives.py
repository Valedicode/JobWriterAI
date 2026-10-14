"""
Contract between decide_tailoring_strategy (step 3) and the writer tools that
consume its directives (steps 4 and 5).

The node-level graph tests stub every tool, so a shape mismatch between what
step 3 emits and what steps 4/5 accept is invisible there. These tests run the
real tools and stub only the LLM.

Run with::

    cd backend
    pytest tests/test_tailoring_directives.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from app.agents.writer_agent import (
    _build_directives_from_report,
    decide_tailoring_strategy,
    rewrite_enhance_content,
    select_prioritize_content,
)
from app.models.schemas import CompatibilityReport, RewrittenContent, SelectedContent


def _report(*, transferable=None, matched=None, missing=None, learnable=None, ats_score=0.9) -> CompatibilityReport:
    return CompatibilityReport.model_validate({
        "aggregate_score": 0.7,
        "level": "high",
        "interpretation": "",
        "dimensions": [
            {"name": "hard_skills", "score": 0.8, "weight": 0.4},
            {"name": "ats_keywords", "score": ats_score, "weight": 0.1},
        ],
        "gap_analysis": {
            "matched_skills": matched or [],
            "transferable_skills": transferable or [],
            "learnable_skills": learnable or [],
            "missing_skills": missing or [],
        },
    })


SPRING_VIA_DJANGO = {
    "required_skill": "Spring Boot",
    "matched_with": "Django",
    "kind": "transferable",
    "transferability": 0.6,
    "rationale": "Both are web frameworks.",
    "bridge_bullet": "Built Spring Boot style services with Django.",
}


def test_transferable_skill_never_ships_drafted_bullet_text():
    directives = _build_directives_from_report(_report(transferable=[SPRING_VIA_DJANGO]))["directives"]
    text = json.dumps(directives)
    assert "inject_bridge_bullet" not in text
    assert SPRING_VIA_DJANGO["bridge_bullet"] not in text
    bridge = next(d for d in directives if d["type"] == "surface_adjacent_experience")
    assert bridge["candidate_skill"] == "Django"
    assert "Do not claim Spring Boot" in bridge["guidance"]


def test_learnable_skill_only_surfaces_the_adjacent_experience():
    learnable = {**SPRING_VIA_DJANGO, "kind": "learnable", "transferability": 0.45, "bridge_bullet": None}
    strategy = _build_directives_from_report(_report(learnable=[learnable]))
    bridge = next(d for d in strategy["directives"] if d["type"] == "surface_adjacent_experience")
    assert bridge["candidate_skill"] == "Django"
    assert "Do not claim Spring Boot" in bridge["guidance"]
    assert not any(d["type"] == "handle_missing_skills" for d in strategy["directives"])
    assert strategy["summary"]["learnable_skills_count"] == 1


def test_transferable_skill_without_candidate_emits_no_directive():
    orphan = {**SPRING_VIA_DJANGO, "matched_with": None}
    strategy = _build_directives_from_report(_report(transferable=[orphan]))
    assert not any(d["type"] == "surface_adjacent_experience" for d in strategy["directives"])
    assert "surface_adjacent_experience" not in strategy["focus_areas"]


def test_low_ats_score_only_asks_for_terms_the_cv_already_has():
    matched = [{"required_skill": "PostgreSQL", "matched_with": "Postgres", "kind": "direct", "transferability": 1.0}]
    missing = [{"required_skill": "Kubernetes", "kind": "missing", "transferability": 0.0}]
    directives = _build_directives_from_report(
        _report(matched=matched, missing=missing, ats_score=0.3)
    )["directives"]
    assert not any(d["type"] == "inject_keywords" for d in directives)
    terms = next(d for d in directives if d["type"] == "use_posting_terms")
    assert terms["skills"] == ["PostgreSQL"]
    assert "Kubernetes" not in json.dumps(terms)


def test_low_ats_score_with_no_matched_skills_emits_no_term_directive():
    missing = [{"required_skill": "Kubernetes", "kind": "missing", "transferability": 0.0}]
    directives = _build_directives_from_report(_report(missing=missing, ats_score=0.1))["directives"]
    assert not any(d["type"] == "use_posting_terms" for d in directives)


# --------------------------------------------------------------------------- #
# Real step-3 output fed to the real step-4/5 tools, LLM stubbed
#
# Regression: the tools joined the directive dicts as strings, which raised
# TypeError before the LLM was called; the blanket except turned that into an
# empty result and the nodes showed a fallback as if the model had answered.
# --------------------------------------------------------------------------- #

class _RecordingLLM:
    """Stands in for chat_model(): records each rendered prompt and
    returns a fixed structured result."""

    def __init__(self, results: dict[type, Any]) -> None:
        self.results = results
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type) -> RunnableLambda:
        def _respond(prompt_value: Any) -> Any:
            self.prompts.append(prompt_value.to_string())
            return self.results[schema]
        return RunnableLambda(_respond)


SELECTED = SelectedContent.model_validate({
    "selected_bullets": [{
        "section": "experience", "original_text": "Built a REST API in Django",
        "relevance_score": 0.9, "reason": "web backend", "bullet_id": "experience.0.responsibilities.0",
    }],
})
REWRITTEN = RewrittenContent.model_validate({
    "rewritten_bullets": [{
        "original": "Built a REST API in Django", "rewritten": "Designed and built a REST API in Django",
        "confidence": 0.8, "bullet_id": "experience.0.responsibilities.0",
    }],
})


def _real_strategy_json() -> str:
    """Strategy exactly as step 3 produces it for a report with a matched and
    a transferable skill - i.e. non-empty dict directives."""
    report = _report(
        matched=[{"required_skill": "Python", "matched_with": "Python", "kind": "direct", "transferability": 1.0}],
        transferable=[SPRING_VIA_DJANGO],
    )
    strategy_json = decide_tailoring_strategy.invoke({
        "compatibility_score_json": report.model_dump_json(),
        "cv_json": "{}",
        "job_json": "{}",
    })
    directives = json.loads(strategy_json)["directives"]
    assert directives and all(isinstance(d, dict) for d in directives)
    return strategy_json


def test_selection_tool_accepts_real_strategy_directives():
    llm = _RecordingLLM({SelectedContent: SELECTED})
    with patch("app.agents.writer_agent.chat_model", return_value=llm):
        result = json.loads(select_prioritize_content.invoke({
            "cv_json": "{}",
            "job_json": "{}",
            "tailoring_strategy_json": _real_strategy_json(),
            "bullet_index_block": "[experience.0.responsibilities.0] Built a REST API in Django",
        }))
    assert "error" not in result
    assert len(result["selected_bullets"]) == 1
    assert "emphasize_skills_section" in llm.prompts[0]
    assert "surface_adjacent_experience" in llm.prompts[0]


def test_rewrite_tool_accepts_real_strategy_directives():
    llm = _RecordingLLM({RewrittenContent: REWRITTEN})
    with patch("app.agents.writer_agent.chat_model", return_value=llm):
        result = json.loads(rewrite_enhance_content.invoke({
            "cv_json": "{}",
            "job_json": "{}",
            "selected_content_json": SELECTED.model_dump_json(),
            "tailoring_strategy_json": _real_strategy_json(),
            "locked_bullets_json": "[]",
        }))
    assert "error" not in result
    assert result["rewritten_bullets"][0]["rewritten"] == "Designed and built a REST API in Django"
    assert "Do not claim Spring Boot" in llm.prompts[0]


def test_tools_report_llm_failure_as_error_not_as_empty_result():
    class _Boom:
        def with_structured_output(self, schema: type) -> RunnableLambda:
            def _raise(_: Any) -> Any:
                raise RuntimeError("provider rejected the schema")
            return RunnableLambda(_raise)

    with patch("app.agents.writer_agent.chat_model", return_value=_Boom()):
        selected = json.loads(select_prioritize_content.invoke({
            "cv_json": "{}", "job_json": "{}", "tailoring_strategy_json": _real_strategy_json(),
        }))
        rewritten = json.loads(rewrite_enhance_content.invoke({
            "cv_json": "{}", "job_json": "{}", "selected_content_json": "{}",
            "tailoring_strategy_json": _real_strategy_json(),
        }))
    assert "provider rejected the schema" in selected["error"]
    assert "provider rejected the schema" in rewritten["error"]
