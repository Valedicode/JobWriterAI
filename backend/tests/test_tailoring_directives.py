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

from app.agents.writer_agent import _build_directives_from_report
from app.models.schemas import CompatibilityReport


def _report(*, transferable=None, matched=None, missing=None, ats_score=0.9) -> CompatibilityReport:
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
