"""
Tests for scoring_agent.py — the standalone compatibility-scoring module.

These tests verify that the extracted module works independently of the Writer
agent's chat workflow.  LLM and embedding calls are mocked so the suite runs
without network access or API keys.

Run with:
    cd backend
    pytest tests/test_compatibility_v2.py -v
"""

import json
from unittest.mock import MagicMock, patch
from typing import List

import pytest

from app.agents.scoring_agent import (
    # Public @tool functions
    calculate_semantic_similarity,
    calculate_bm25_score,
    calculate_compatibility_score_v2,
    calculate_compatibility_score,
    match_skill_pairs,
    assess_transferability_llm,
    build_gap_analysis,
    # Internal function used by decide_tailoring_strategy fallback
    _calculate_compatibility_score_v2_internal,
    # Internal helpers exercised directly for OR-alternative group tests
    _collapse_required_skill_groups,
    _score_hard_skills,
    # Constants
    MATCHED_THRESHOLD,
    TRANSFERABLE_THRESHOLD,
)
from app.models.schemas import (
    CompatibilityReport,
    GapAnalysis,
    SkillMatch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CV = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "000-000-0000",
    "skills": ["python", "fastapi", "react"],
    "education": ["BSc Computer Science"],
    "experience": [
        {
            "position": "Senior Software Engineer",
            "company": "Acme Corp",
            "duration": "2020-2024",
            "responsibilities": ["Built REST APIs with FastAPI and Python"],
        }
    ],
    "projects": [],
}

MINIMAL_JOB = {
    "job_title": "Software Engineer",
    "job_level": "senior",
    "required_skills": ["python", "fastapi"],
    "preferred_skills": ["react"],
    "years_experience": 3,
    "employment_type": "Full-time",
    "location": "Remote",
    "responsibilities": ["Build backend services"],
    "qualifications": [],
    "key_requirements": ["python"],
}

FAMILY_JOB = {
    **MINIMAL_JOB,
    "required_skills": ["vue"],  # candidate has react — same family
    "preferred_skills": [],
}


def _fake_embedding(text: str) -> List[float]:
    """Return a deterministic unit vector for testing."""
    return [1.0] + [0.0] * 1535


# ---------------------------------------------------------------------------
# Test 1: calculate_compatibility_score_v2 returns a valid CompatibilityReport
# ---------------------------------------------------------------------------

def test_calculate_compatibility_score_v2_returns_report():
    """Call the tool with mocked embeddings; result must be a valid CompatibilityReport."""
    with patch("app.agents.scoring_agent.get_embeddings", return_value=[[1.0] + [0.0] * 1535]):
        raw = calculate_compatibility_score_v2.invoke({
            "cv_json": json.dumps(MINIMAL_CV),
            "job_json": json.dumps(MINIMAL_JOB),
        })

    report = CompatibilityReport.model_validate_json(raw)
    assert 0.0 <= report.aggregate_score <= 1.0
    assert report.level in ("low", "medium", "high", "excellent", "unknown")
    assert len(report.dimensions) == 5
    dim_names = {d.name for d in report.dimensions}
    assert dim_names == {"hard_skills", "experience", "seniority", "domain", "ats_keywords"}


# ---------------------------------------------------------------------------
# Test 2: match_skill_pairs — direct match
# ---------------------------------------------------------------------------

def test_match_skill_pairs_direct_match():
    """When the CV lists the required skill exactly, expect kind='direct' with transferability=1.0."""
    raw = match_skill_pairs.invoke({
        "cv_json": json.dumps(MINIMAL_CV),
        "job_json": json.dumps(MINIMAL_JOB),
    })

    matches = json.loads(raw)
    # python is in both CV skills and required_skills
    python_match = next((m for m in matches if m["required_skill"] == "python"), None)
    assert python_match is not None
    assert python_match["kind"] == "direct"
    assert python_match["transferability"] == 1.0


# ---------------------------------------------------------------------------
# Test 3: match_skill_pairs — family match (react -> vue, same skill-graph family)
# ---------------------------------------------------------------------------

def test_match_skill_pairs_family_match():
    """When the CV has 'react' and the job requires 'vue', expect kind='family'."""
    with patch("app.agents.scoring_agent.get_embeddings", return_value=[[1.0] + [0.0] * 1535]):
        raw = match_skill_pairs.invoke({
            "cv_json": json.dumps(MINIMAL_CV),
            "job_json": json.dumps(FAMILY_JOB),
        })

    matches = json.loads(raw)
    vue_match = next((m for m in matches if m["required_skill"] == "vue"), None)
    assert vue_match is not None
    assert vue_match["kind"] == "family"
    assert vue_match["transferability"] > 0.5


# ---------------------------------------------------------------------------
# Test 4: build_gap_analysis — correct bucketing of SkillMatch entries
# ---------------------------------------------------------------------------

def test_build_gap_analysis_buckets():
    """Hand-craft SkillMatch entries and verify they land in the right buckets."""
    skill_matches = [
        {
            "required_skill": "python",
            "matched_with": "python",
            "kind": "direct",
            "transferability": 1.0,
            "rationale": "direct",
            "bridge_bullet": None,
            "is_required": True,
        },
        {
            "required_skill": "vue",
            "matched_with": "react",
            "kind": "family",
            "transferability": 0.85,  # above TRANSFERABLE_THRESHOLD but below MATCHED_THRESHOLD
            "rationale": "same family",
            "bridge_bullet": None,
            "is_required": True,
        },
        {
            "required_skill": "kubernetes",
            "matched_with": None,
            "kind": "missing",
            "transferability": 0.1,  # below TRANSFERABLE_THRESHOLD
            "rationale": "no match",
            "bridge_bullet": None,
            "is_required": True,
        },
    ]

    raw = build_gap_analysis.invoke({
        "skill_matches_json": json.dumps(skill_matches),
        "cv_json": json.dumps(MINIMAL_CV),
        "job_json": json.dumps(MINIMAL_JOB),
    })

    gap = GapAnalysis.model_validate_json(raw)
    matched_skills = [m.required_skill for m in gap.matched_skills]
    transferable_skills = [m.required_skill for m in gap.transferable_skills]
    missing_skills = [m.required_skill for m in gap.missing_skills]

    assert "python" in matched_skills          # transferability=1.0 >= MATCHED_THRESHOLD
    assert "vue" in transferable_skills        # 0.55 <= 0.85 < 0.95
    assert "kubernetes" in missing_skills      # transferability=0.1 < TRANSFERABLE_THRESHOLD


# ---------------------------------------------------------------------------
# Test 5: calculate_semantic_similarity — structure check with mocked embeddings
# ---------------------------------------------------------------------------

def test_calculate_semantic_similarity_structure():
    """Mocked embeddings; result must have similarity_score in [0, 1]."""
    fake_emb = [[1.0] + [0.0] * 1535]
    with patch("app.agents.scoring_agent.get_embeddings", return_value=fake_emb):
        raw = calculate_semantic_similarity.invoke({
            "resume_text": "Python developer with FastAPI experience",
            "job_text": "Looking for a Python backend engineer",
        })

    result = json.loads(raw)
    assert "similarity_score" in result
    assert 0.0 <= result["similarity_score"] <= 1.0
    assert result.get("method") == "cosine_similarity"


# ---------------------------------------------------------------------------
# Test 6: calculate_bm25_score — structure check
# ---------------------------------------------------------------------------

def test_calculate_bm25_score_structure():
    """BM25 result must contain bm25_scores list and average_score."""
    sections = json.dumps(["Python and FastAPI experience", "React frontend work"])
    raw = calculate_bm25_score.invoke({
        "resume_sections": sections,
        "job_requirements": "senior python fastapi engineer",
    })

    result = json.loads(raw)
    assert "bm25_scores" in result
    assert isinstance(result["bm25_scores"], list)
    assert len(result["bm25_scores"]) == 2
    assert "average_score" in result
    assert 0.0 <= result["average_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test 7: calculate_compatibility_score_v2 — invalid JSON returns level='unknown'
# ---------------------------------------------------------------------------

def test_compatibility_score_v2_invalid_json():
    """Garbage input must return a CompatibilityReport with level='unknown', not raise."""
    raw = calculate_compatibility_score_v2.invoke({
        "cv_json": "not valid json }{",
        "job_json": '{"job_title": "Engineer"}',
    })

    report = CompatibilityReport.model_validate_json(raw)
    assert report.level == "unknown"
    assert report.aggregate_score == 0.0
    assert len(report.warnings) > 0


# ---------------------------------------------------------------------------
# Test 8: writer_agent's tools list still contains all 8 scoring tool names
# ---------------------------------------------------------------------------

def test_writer_agent_scoring_tools_importable():
    """
    Each scoring tool is still importable as a module-level @tool from
    ``app.agents.writer_agent``. The legacy ``agent`` (a create_agent
    instance) was retired during the orchestrator migration; the orchestrator
    invokes these tools directly, so this test now checks the public surface
    rather than the create_agent binding.
    """
    expected_scoring_tools = {
        "calculate_semantic_similarity",
        "calculate_bm25_score",
        "calculate_compatibility_score",
        "match_skill_pairs",
        "assess_transferability_llm",
        "calculate_compatibility_score_v2",
        "build_gap_analysis",
        "decide_tailoring_strategy",
    }

    import app.agents.writer_agent as writer_agent  # noqa: PLC0415

    missing = set()
    for name in expected_scoring_tools:
        if not hasattr(writer_agent, name):
            missing.add(name)
            continue
        attr = getattr(writer_agent, name)
        # Either a bare callable (plain function) or a LangChain tool object
        # exposing .invoke() / .run() is acceptable.
        if not (callable(attr) or hasattr(attr, "invoke") or hasattr(attr, "run")):
            missing.add(name)
    assert not missing, f"Missing tools in writer_agent module: {missing}"


# ---------------------------------------------------------------------------
# OR-alternative requirement groups (required_skill_groups)
#
# Regression for a real posting phrased "Java and WildFly/Quarkus/Spring Boot
# ... was Du noch nicht kennst, lernst Du bei uns" - a candidate with Java and
# Spring Boot only (missing WildFly/Quarkus) fully satisfies this, but a flat
# required_skills list scores it as two genuine gaps.
# ---------------------------------------------------------------------------

def _skill_match(name: str, kind: str, transferability: float, matched_with=None, is_required=True):
    return SkillMatch(
        required_skill=name,
        matched_with=matched_with,
        kind=kind,
        transferability=transferability,
        rationale=f"test fixture for {name}",
        bridge_bullet=None,
        is_required=is_required,
    )


def test_collapse_required_skill_groups_keeps_best_alternative_only():
    matches = [
        _skill_match("Java", "direct", 1.0, matched_with="Java"),
        _skill_match("WildFly", "missing", 0.0),
        _skill_match("Quarkus", "missing", 0.0),
        _skill_match("Spring Boot", "direct", 1.0, matched_with="Spring Boot"),
    ]
    groups = [["Java"], ["WildFly", "Quarkus", "Spring Boot"]]

    collapsed = _collapse_required_skill_groups(matches, groups)

    # Two requirements in, two requirements out - not four.
    assert len(collapsed) == 2
    by_name = {m.required_skill: m for m in collapsed}
    assert by_name["Java"].transferability == 1.0
    # The JVM-framework group is represented by its best alternative
    # (Spring Boot, matched), not diluted by the two missing ones.
    jvm_group = by_name["WildFly / Quarkus / Spring Boot"]
    assert jvm_group.kind == "direct"
    assert jvm_group.transferability == 1.0
    assert jvm_group.matched_with == "Spring Boot"


def test_collapse_required_skill_groups_reports_missing_when_all_alternatives_missing():
    """If genuinely none of the alternatives are covered, the group must
    still show up as a real gap - collapsing must not manufacture a match."""
    matches = [
        _skill_match("Java", "direct", 1.0, matched_with="Java"),
        _skill_match("WildFly", "missing", 0.0),
        _skill_match("Quarkus", "missing", 0.0),
        _skill_match("Spring Boot", "missing", 0.0),
    ]
    groups = [["Java"], ["WildFly", "Quarkus", "Spring Boot"]]

    collapsed = _collapse_required_skill_groups(matches, groups)

    by_name = {m.required_skill: m for m in collapsed}
    assert len(collapsed) == 2
    assert by_name["WildFly / Quarkus / Spring Boot"].kind == "missing"


def test_collapse_required_skill_groups_leaves_ungrouped_required_skills_untouched():
    matches = [
        _skill_match("Java", "direct", 1.0, matched_with="Java"),
        _skill_match("Docker", "missing", 0.0),
    ]
    # Only Java is covered by a group; Docker has no stated alternatives.
    collapsed = _collapse_required_skill_groups(matches, [["Java"]])
    by_name = {m.required_skill: m for m in collapsed}
    assert "Docker" in by_name
    assert by_name["Docker"].kind == "missing"


def test_hard_skills_score_not_penalized_by_or_group_alternatives():
    """The actual bug: without grouping, two 'missing' alternatives drag the
    weighted hard-skills average down even though the requirement (any ONE
    JVM framework) is fully satisfied by Spring Boot."""
    ungrouped = [
        _skill_match("Java", "direct", 1.0, matched_with="Java"),
        _skill_match("WildFly", "missing", 0.0),
        _skill_match("Quarkus", "missing", 0.0),
        _skill_match("Spring Boot", "direct", 1.0, matched_with="Spring Boot"),
    ]
    grouped = _collapse_required_skill_groups(ungrouped, [["Java"], ["WildFly", "Quarkus", "Spring Boot"]])

    ungrouped_score, _ = _score_hard_skills(ungrouped)
    grouped_score, _ = _score_hard_skills(grouped)

    assert ungrouped_score == pytest.approx(0.5)  # (1.0 + 0 + 0 + 1.0) / 4
    assert grouped_score == pytest.approx(1.0)     # (1.0 + 1.0) / 2 - fully satisfied
    assert grouped_score > ungrouped_score


def test_match_skill_pairs_applies_required_skill_groups_end_to_end():
    """Full match_skill_pairs cascade, with the embedding/LLM fallbacks
    stubbed out so WildFly/Quarkus deterministically stay 'missing' pre-group
    -collapse, proving the collapse happens on real cascade output too."""
    cv = {
        **MINIMAL_CV,
        "skills": ["Java", "Spring Boot"],
        "experience": [
            {
                "position": "Backend Engineer",
                "company": "Acme Corp",
                "responsibilities": ["Built services with Java and Spring Boot"],
            }
        ],
    }
    job = {
        **MINIMAL_JOB,
        "required_skills": ["Java", "WildFly", "Quarkus", "Spring Boot"],
        "required_skill_groups": [["Java"], ["WildFly", "Quarkus", "Spring Boot"]],
        "preferred_skills": [],
    }

    with patch("app.agents.scoring_agent._embedding_transferability", return_value=None), \
         patch("app.agents.scoring_agent._assess_transferability_llm", return_value={}):
        raw = match_skill_pairs.invoke({"cv_json": json.dumps(cv), "job_json": json.dumps(job)})

    matches = json.loads(raw)
    assert len(matches) == 2
    by_name = {m["required_skill"]: m for m in matches}
    assert by_name["Java"]["kind"] == "direct"
    jvm_group = by_name["WildFly / Quarkus / Spring Boot"]
    assert jvm_group["kind"] == "direct"
    assert jvm_group["matched_with"] == "Spring Boot"
