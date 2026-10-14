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


# ---------------------------------------------------------------------------
# Required vs preferred asymmetry, eligibility, and matching edge cases
#
# Regressions for a Werkstudent posting (German + English + enrolled CS
# student = must-have; Java/Spring/Quarkus/WildFly = strongly preferred).
# Before these fixes: nice-to-haves were averaged in as near-penalties, the
# language / enrollment must-haves were never scored, project technologies
# were invisible to matching, and "C++" / "Node.js" could not be found in
# free text.
# ---------------------------------------------------------------------------

from app.agents.scoring_agent import (  # noqa: E402
    _check_eligibility,
    _detect_direct_match,
    _estimate_candidate_years,
    _level_bucket,
    _score_ats_keywords,
    _seniority_to_score,
)

STUDENT_CV = {
    **MINIMAL_CV,
    "skills": ["Python", "Git"],
    "education": [
        {
            "institution": "RWTH Aachen",
            "degree": "B.Sc. Computer Science",
            "dates": "Expected: September 2027",
        }
    ],
    "experience": [],
    "projects": [
        {
            "name": "Campus API",
            "description": "REST backend for a student club",
            "technologies": ["Java", "Spring Boot"],
            "outcomes": ["Deployed with Docker"],
        }
    ],
    "languages": [
        {"language": "German", "fluency": "native"},
        {"language": "English", "fluency": "C1"},
    ],
}

STUDENT_JOB = {
    **MINIMAL_JOB,
    "job_title": "Werkstudent Software Engineering (m/w/d)",
    "job_level": "Werkstudent",
    "required_skills": ["Java"],
    "preferred_skills": ["Spring Boot", "Quarkus", "WildFly", "Kubernetes", "Kafka", "Terraform"],
    "required_languages": ["German (C1)", "English (fluent)"],
    "education_requirements": ["Enrolled student in Computer Science or a related field"],
    "years_experience": None,
}


def test_hard_skills_missing_must_have_is_not_rescued_by_nice_to_haves():
    """Missing the single required skill must score far below having it,
    regardless of how many nice-to-haves the candidate collects."""
    missing_java = [_skill_match("Java", "missing", 0.0)] + [
        _skill_match(f"pref{i}", "direct", 1.0, matched_with=f"pref{i}", is_required=False) for i in range(5)
    ]
    has_java = [_skill_match("Java", "direct", 1.0, matched_with="Java")] + [
        _skill_match(f"pref{i}", "missing", 0.0, is_required=False) for i in range(6)
    ]
    missing_score, _ = _score_hard_skills(missing_java, "Werkstudent")
    has_score, rationale = _score_hard_skills(has_java, "Werkstudent")

    assert missing_score <= 0.2
    assert has_score >= 0.8
    assert "student-level" in rationale


def test_hard_skills_preferred_weight_scales_with_level():
    """The same nice-to-have gaps cost a senior applicant more than a student."""
    matches = [_skill_match("Java", "direct", 1.0, matched_with="Java")] + [
        _skill_match(f"pref{i}", "missing", 0.0, is_required=False) for i in range(4)
    ]
    student, _ = _score_hard_skills(matches, "Werkstudent")
    senior, _ = _score_hard_skills(matches, "Senior Engineer")
    assert student > senior
    assert _level_bucket("Werkstudent (m/w/d)") == "student"
    assert _level_bucket("Senior Backend Engineer") == "senior"
    assert _level_bucket("") == "mid"


def test_direct_match_reads_project_technologies_and_symbol_names():
    cv = {
        **MINIMAL_CV,
        "skills": ["Python"],
        "experience": [
            {"position": "Developer", "company": "X", "responsibilities": ["Built tooling in C++ and Node.js"]}
        ],
        "projects": [{"name": "P", "description": "app", "technologies": ["Java", "Spring Boot"]}],
    }
    assert _detect_direct_match("Java", cv) is not None
    assert _detect_direct_match("Spring Boot", cv) is not None
    assert _detect_direct_match("C++", cv) is not None
    assert _detect_direct_match("Node.js", cv) is not None
    # "java" must not match inside "javascript", "c" must not match inside "c++"
    js_only = {**MINIMAL_CV, "skills": ["JavaScript"], "experience": [], "projects": []}
    cpp_only = {**MINIMAL_CV, "skills": ["C++"], "experience": [], "projects": []}
    assert _detect_direct_match("Java", js_only) is None
    assert _detect_direct_match("C", cpp_only) is None


def test_eligibility_checks_languages_and_enrollment():
    checks = {c.requirement: c for c in _check_eligibility(STUDENT_CV, STUDENT_JOB)}
    assert checks["German (C1)"].status == "met"
    assert checks["English (fluent)"].status == "met"
    assert checks["Enrolled student in Computer Science or a related field"].status == "met"

    # Level below requirement is unmet; a missing languages section is unclear, not unmet.
    b1_cv = {**STUDENT_CV, "languages": [{"language": "German", "fluency": "B1"}]}
    b1 = {c.requirement: c for c in _check_eligibility(b1_cv, STUDENT_JOB)}
    assert b1["German (C1)"].status == "unmet"
    assert b1["English (fluent)"].status == "unmet"

    no_lang = {c.requirement: c for c in _check_eligibility({**STUDENT_CV, "languages": []}, STUDENT_JOB)}
    assert no_lang["German (C1)"].status == "unclear"

    wrong_field = {
        **STUDENT_CV,
        "education": [{"degree": "B.A. History", "institution": "Uni", "dates": "Expected: 2027"}],
    }
    wf = {c.requirement: c for c in _check_eligibility(wrong_field, STUDENT_JOB)}
    assert wf["Enrolled student in Computer Science or a related field"].status == "unmet"


def test_misfiled_language_in_required_skills_is_routed_to_eligibility():
    """If extraction drops "German" into required_skills it must not become an
    unbridgeable "missing" technical skill."""
    job = {**STUDENT_JOB, "required_skills": ["Java", "German"], "required_languages": []}
    with patch("app.agents.scoring_agent._embedding_transferability", return_value=None), \
         patch("app.agents.scoring_agent._assess_transferability_llm", return_value={}):
        matches = json.loads(match_skill_pairs.invoke({"cv_json": json.dumps(STUDENT_CV), "job_json": json.dumps(job)}))
    assert [m["required_skill"] for m in matches if m["is_required"]] == ["Java"]
    checks = _check_eligibility(STUDENT_CV, job)
    assert any(c.kind == "language" and c.status == "met" for c in checks)


def test_v2_report_for_student_posting_end_to_end():
    # Embedding fallback stubbed out: the fake embedding is identical for every
    # text, which would otherwise make every nice-to-have "transferable".
    with patch("app.agents.scoring_agent.get_embeddings", return_value=[[1.0] + [0.0] * 1535]), \
         patch("app.agents.scoring_agent._embedding_transferability", return_value=None), \
         patch("app.agents.scoring_agent._assess_transferability_llm", return_value={}):
        raw = calculate_compatibility_score_v2.invoke({
            "cv_json": json.dumps(STUDENT_CV),
            "job_json": json.dumps(STUDENT_JOB),
        })
    report = CompatibilityReport.model_validate_json(raw)
    dims = {d.name: d for d in report.dimensions}
    assert "eligibility" in dims and dims["eligibility"].score == pytest.approx(1.0)
    assert dims["hard_skills"].score >= 0.8  # Java found in project technologies; nice-to-haves barely count
    assert report.level in ("high", "excellent")
    assert not report.warnings
    # Missing nice-to-haves are named as such, not as required gaps
    assert "Genuine gaps on required skills" not in report.interpretation
    assert "Nice-to-haves not on the CV" in report.interpretation
    assert len(report.gap_analysis.eligibility) == 3


def test_v2_unmet_eligibility_caps_level_and_warns():
    cv = {**STUDENT_CV, "languages": [{"language": "English", "fluency": "C1"}]}
    with patch("app.agents.scoring_agent.get_embeddings", return_value=[[1.0] + [0.0] * 1535]), \
         patch("app.agents.scoring_agent._assess_transferability_llm", return_value={}):
        raw = calculate_compatibility_score_v2.invoke({"cv_json": json.dumps(cv), "job_json": json.dumps(STUDENT_JOB)})
    report = CompatibilityReport.model_validate_json(raw)
    assert report.level in ("low", "medium")
    assert any("German (C1)" in w for w in report.warnings)
    assert "Eligibility not met: German (C1)" in report.interpretation


def test_eligibility_dimension_omitted_when_posting_states_none():
    with patch("app.agents.scoring_agent.get_embeddings", return_value=[[1.0] + [0.0] * 1535]):
        raw = calculate_compatibility_score_v2.invoke({"cv_json": json.dumps(MINIMAL_CV), "job_json": json.dumps(MINIMAL_JOB)})
    report = CompatibilityReport.model_validate_json(raw)
    assert "eligibility" not in {d.name for d in report.dimensions}


def test_seniority_uses_whole_words_and_knows_students():
    assert _seniority_to_score("Internal Tools Developer") is None
    assert _seniority_to_score("Software Engineering Intern") == 0
    assert _seniority_to_score("Werkstudent Backend") == 0
    assert _seniority_to_score("Team Leadership Coach") is None
    assert _seniority_to_score("Senior Staff Engineer") == 5


def test_candidate_years_ignores_unrelated_numbers_and_merges_overlaps():
    cv = {
        **MINIMAL_CV,
        "experience": [
            {"position": "Dev", "company": "A", "duration": "2020 - 2023",
             "responsibilities": ["Maintained a 10 years old codebase"]},
            {"position": "Dev", "company": "B", "duration": "2022 - 2024", "responsibilities": []},
        ],
    }
    assert _estimate_candidate_years(cv) == pytest.approx(4.0)  # 2020-2024 merged, not 3 + 2
    explicit = {**MINIMAL_CV, "experience": [{"position": "Dev", "company": "A",
                                                "responsibilities": ["5+ years of professional experience"]}]}
    assert _estimate_candidate_years(explicit) == pytest.approx(5.0)


def test_ats_score_is_verbatim_term_coverage():
    score, rationale = _score_ats_keywords(STUDENT_CV, STUDENT_JOB)
    # Java + Spring Boot present verbatim; Quarkus, WildFly, Kubernetes, Kafka, Terraform absent
    assert score == pytest.approx(2 / 7)
    assert "Quarkus" in rationale


@pytest.mark.parametrize(
    ("duration", "years"),
    [
        ("2022 - 2024", 2.0),                        # year-only keeps its old meaning
        ("March 2024 -- August 2025", 1.5),          # the schema's own "--" format; end month included
        ("Oct 2023 -- Jan 2026", 2 + 4 / 12),
        ("10/2023 – 01/2026", 2 + 4 / 12),      # numeric months, en dash
        ("03.2021 - 12.2022", 1 + 10 / 12),
        ("Sept. 2022 to June 2023", 10 / 12),
        ("März 2021 – Dezember 2022", 1 + 10 / 12),  # German month names
        ("from 2020 to 2022", 2.0),                  # "from" is not read as a month
    ],
)
def test_candidate_years_reads_month_year_ranges(duration, years):
    cv = {**MINIMAL_CV, "experience": [{"position": "Dev", "company": "A", "duration": duration}]}
    assert _estimate_candidate_years(cv) == pytest.approx(years, abs=1e-6)


def test_candidate_years_open_and_future_ends_count_until_today():
    from datetime import datetime

    now = datetime.now()
    months_since_2020 = (now.year - 2020) * 12 + now.month - 1
    for duration in ("Januar 2020 bis heute", "Jan 2020 -- present", "seit Januar 2020"):
        cv = {**MINIMAL_CV, "experience": [{"position": "Dev", "company": "A", "duration": duration}]}
        assert _estimate_candidate_years(cv) == pytest.approx(months_since_2020 / 12), duration

    # A contract that ends in the future counts up to today, not zero.
    future = {**MINIMAL_CV, "experience": [{"position": "Dev", "company": "A",
                                            "duration": f"Jan 2020 -- Dec {now.year + 3}"}]}
    assert _estimate_candidate_years(future) == pytest.approx(months_since_2020 / 12)
    # A range that has not started yet counts nothing.
    upcoming = {**MINIMAL_CV, "experience": [{"position": "Dev", "company": "A",
                                              "duration": f"{now.year + 1} - {now.year + 2}"}]}
    assert _estimate_candidate_years(upcoming) == 0.0
