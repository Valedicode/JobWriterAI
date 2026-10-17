"""
Requirement rubric: each requirement of the posting, its status, and the CV
facts and bullets that show it.

- rows come from the scorer's matches (status by bucket, category by where the
  posting lists the skill, OR-groups as alternatives), must-haves first;
- evidence is found by skill-graph entity or by a plain mention;
- knockouts (eligibility and the posting's stated conditions) are pass/fail
  and stay out of the weighted score; an unmet one still caps the level;
- after tailoring, the evidence is searched again and the alignment delta is
  reported, and bullet selection gets the evidence map.

Run with::

    cd backend
    pytest tests/test_requirement_rubric.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langgraph.types import Command

from app.agents import scoring_agent
from app.models.schemas import EligibilityCheck, RequirementRow, SkillMatch
from app.services.bullet_index import index_cv_bullets
from app.services.fact_ledger import build_fact_ledger
from app.services.requirement_rubric import (
    UNEVIDENCED_FACTOR,
    evidence_prompt_block,
    find_evidence,
    knockout_rows,
    refresh_evidence,
    rubric_alignment,
    skill_rows,
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

RCV = {
    "skills": ["Python", "Docker", "REST APIs"],
    "experience": [{
        "position": "Engineer", "company": "X", "duration": "2021 - 2023",
        "responsibilities": ["Ran services on k8s", "Designed REST APIs for partners", "Wrote the release notes"],
    }],
}
RLEDGER = build_fact_ledger(RCV, index_cv_bullets(RCV))


def _match(skill, kind="direct", t=1.0, required=True, matched_with=None, scored_by="deterministic"):
    return SkillMatch(required_skill=skill, matched_with=matched_with, kind=kind, transferability=t,
                      rationale=f"{skill} rationale", is_required=required, scored_by=scored_by)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def test_evidence_by_entity_finds_aliases():
    fact_ids, bullet_ids = find_evidence(["Kubernetes"], RLEDGER)
    assert bullet_ids == ["experience.0.responsibilities.0"]
    assert fact_ids == ["experience.0.responsibilities.0"]


def test_evidence_by_mention_for_skills_the_graph_does_not_know():
    fact_ids, bullet_ids = find_evidence(["REST APIs"], RLEDGER)
    assert "experience.0.responsibilities.1" in bullet_ids
    assert any(f.startswith("skill.") for f in fact_ids)


def test_dates_and_titles_are_not_evidence():
    assert find_evidence(["2021"], RLEDGER) == ([], [])


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

def test_skill_rows_status_category_order_and_ids():
    matches = [
        _match("Kubernetes", required=False, matched_with="k8s"),
        _match("Spring Boot / Quarkus", kind="learnable", t=0.45, matched_with="Django", scored_by="llm_judge"),
        _match("Kafka", kind="missing", t=0.0, required=False),
        _match("Python", matched_with="Python"),
        _match("Go", kind="missing", t=0.0),
    ]
    buckets = ["matched", "learnable", "missing", "matched", "missing"]
    job = {"required_skill_groups": [["Spring Boot", "Quarkus"]], "learnable_skills": ["Kafka"]}
    rows = skill_rows(matches, buckets, job, RLEDGER, judge_model="judge/model")

    assert [(r.requirement_id, r.text, r.category, r.status) for r in rows] == [
        ("req.0", "Spring Boot / Quarkus", "must_have", "learnable_gap"),
        ("req.1", "Python", "must_have", "met"),
        ("req.2", "Go", "must_have", "missing"),
        ("req.3", "Kubernetes", "nice_to_have", "met"),
        ("req.4", "Kafka", "learnable", "learnable_gap"),  # the employer teaches it
    ]
    spring = rows[0]
    assert spring.alternatives == ["Spring Boot", "Quarkus"]
    assert spring.matched_with == "Django" and spring.scored_by == "llm_judge" and spring.judge_model == "judge/model"
    assert rows[1].judge_model is None
    assert rows[2].matched_with is None and rows[2].evidence_fact_ids == []
    assert rows[3].evidence_bullet_ids == ["experience.0.responsibilities.0"]


def test_knockout_rows_cover_eligibility_and_stated_conditions():
    checks = [
        EligibilityCheck(requirement="German (C1)", kind="language", status="met", matched_with="German C1"),
        EligibilityCheck(requirement="English (fluent)", kind="language", status="unmet"),
        EligibilityCheck(requirement="Enrolled student", kind="education", status="unclear"),
    ]
    rows = knockout_rows(checks, {"work_mode": "hybrid", "work_permit": "EU work permit", "hours_per_week": ""})
    assert [(r.requirement_id, r.text, r.status) for r in rows] == [
        ("ko.0", "German (C1)", "met"),
        ("ko.1", "English (fluent)", "missing"),
        ("ko.2", "Enrolled student", "unclear"),
        ("ko.3", "Work mode: hybrid", "unclear"),
        ("ko.4", "Work permit: EU work permit", "unclear"),
    ]
    assert all(r.category == "knockout" for r in rows)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def _row(category, status, bullets=()):
    return RequirementRow(requirement_id="req.0", text="x", category=category, status=status,
                          evidence_bullet_ids=list(bullets))


def test_alignment_weighs_must_haves_double_and_rewards_bullet_evidence():
    rows = [_row("must_have", "met", ["b1"]), _row("nice_to_have", "missing")]
    assert rubric_alignment(rows) == pytest.approx(2 / 3, abs=1e-4)
    unevidenced = [_row("must_have", "met"), _row("nice_to_have", "missing")]
    assert rubric_alignment(unevidenced) == pytest.approx(2 * UNEVIDENCED_FACTOR / 3, abs=1e-4)


def test_alignment_ignores_knockouts_learnables_and_unclear():
    assert rubric_alignment([_row("knockout", "met"), _row("learnable", "learnable_gap")]) is None
    assert rubric_alignment([]) is None


def test_refresh_evidence_finds_a_skill_newly_shown_in_a_bullet():
    rows = skill_rows([_match("Python", matched_with="Python")], ["matched"], {}, RLEDGER)
    assert rows[0].evidence_bullet_ids == []
    tailored = json.loads(json.dumps(RCV))
    tailored["experience"][0]["responsibilities"][2] = "Wrote the release notes tooling in Python"
    after = refresh_evidence(rows, build_fact_ledger(tailored, index_cv_bullets(tailored)))
    assert after[0].evidence_bullet_ids == ["experience.0.responsibilities.2"]
    assert after[0].status == "met"
    assert rubric_alignment(after) > rubric_alignment(rows)


def test_evidence_prompt_block_lists_skill_rows_only():
    rows = [r.model_dump() for r in skill_rows([_match("Kubernetes", matched_with="k8s")], ["matched"], {}, RLEDGER)]
    rows += [r.model_dump() for r in knockout_rows([], {"work_mode": "remote"})]
    assert evidence_prompt_block(rows) == "[req.0] must_have Kubernetes (met): experience.0.responsibilities.0"


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #

SCV = {"skills": ["Python"], "languages": [], "experience": [
    {"position": "Engineer", "company": "X", "duration": "2020 - 2024", "responsibilities": ["Built APIs in Python"]},
]}
SJOB = {"job_title": "Engineer", "job_level": "mid", "required_skills": ["Python", "Go"],
        "required_languages": ["German (C1)"], "work_mode": "onsite",
        "employment_type": "x", "location": "Berlin", "responsibilities": [], "key_requirements": []}


@pytest.fixture
def offline_scorer():
    with patch.object(scoring_agent, "_embedding_transferability", return_value=None), \
         patch.object(scoring_agent, "_assess_transferability_llm", return_value={}), \
         patch.object(scoring_agent, "_score_domain", return_value=(0.5, "stub")):
        yield


def _score(cv, job=SJOB):
    return scoring_agent._calculate_compatibility_score_v2_internal(cv, job)


def test_report_carries_the_rubric_and_knockouts(offline_scorer):
    report = _score(SCV)
    assert [(r.text, r.status) for r in report.rubric] == [("Python", "met"), ("Go", "missing")]
    assert report.rubric[0].evidence_bullet_ids == ["experience.0.responsibilities.0"]
    assert [(r.text, r.status) for r in report.knockouts] == [("German (C1)", "unclear"), ("Work mode: onsite", "unclear")]
    assert {m.scored_by for m in report.gap_analysis.matched_skills} == {"deterministic"}


def test_knockouts_stay_out_of_the_weighted_score(offline_scorer):
    met = _score({**SCV, "languages": [{"language": "German", "fluency": "C1"}]})
    unmet = _score({**SCV, "languages": [{"language": "German", "fluency": "A2"}]})
    assert met.aggregate_score == unmet.aggregate_score
    elig = next(d for d in met.dimensions if d.name == "eligibility")
    assert elig.weight == 0.0 and elig.score == 1.0
    # An unmet knockout still caps the level when the rest would rate high.
    assert unmet.knockouts[0].status == "missing"


def test_unmet_knockout_caps_a_high_level(offline_scorer):
    job = {**SJOB, "required_skills": ["Python"]}
    with patch.object(scoring_agent, "_score_domain", return_value=(1.0, "stub")):
        unmet = _score({**SCV, "languages": [{"language": "German", "fluency": "A2"}]}, job)
    assert unmet.level == "medium"
    assert any(w.startswith("Level capped") for w in unmet.warnings)


def test_rescoring_eligibility_updates_knockouts_and_keeps_the_rubric(offline_scorer):
    report = _score(SCV)
    cv = {**SCV, "languages": [{"language": "German", "fluency": "C1"}]}
    rescored = scoring_agent.rescore_eligibility(report, cv, SJOB)
    assert rescored.knockouts[0].status == "met"
    assert rescored.rubric == report.rubric


def test_family_and_judge_matches_record_how_they_were_decided(offline_scorer):
    family = scoring_agent._match_skill_pairs_internal(
        {"skills": ["Django"]}, {"required_skills": ["Flask"]}
    )
    assert family[0].kind == "family" and family[0].scored_by == "skill_graph"
    judged = scoring_agent._match_from_verdict(
        _match("Spring Boot", kind="missing", t=0.0),
        scoring_agent._LLMTransferabilityItem(required_skill="Spring Boot", best_candidate_skill="Django",
                                              verdict="adjacent", rationale="r"),
    )
    assert judged.scored_by == "llm_judge"


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

BULLET = "experience.0.responsibilities.0"  # "Designed the first algorithm"
REPORT_WITH_RUBRIC = {
    **COMPATIBILITY_REPORT,
    "rubric": [RequirementRow(requirement_id="req.0", text="Python", category="must_have", status="met",
                              matched_with="Python", evidence_fact_ids=["skill.python"]).model_dump()],
    "knockouts": [],
}
SELECTED = {"selected_bullets": [{"section": "experience", "original_text": "Designed the first algorithm",
                                  "relevance_score": 0.9, "reason": "", "bullet_id": BULLET}],
            "section_order": [], "sections_to_emphasize": [], "items_to_de_emphasize": []}
REWRITE = {"rewritten_bullets": [{"original": "Designed the first algorithm",
                                  "rewritten": "Designed the first algorithm in Python",
                                  "confidence": 0.9, "keywords_added": ["Python"], "bullet_id": BULLET,
                                  "cited_fact_ids": ["skill.python"]}],
           "updated_summary": "", "keywords_inserted": ["Python"]}


def test_rubric_flows_through_selection_review_and_assembly(graph, patched_tools):  # noqa: F811
    seen: dict = {}

    def select(kwargs):
        seen["evidence"] = kwargs["evidence_block"]
        return json.dumps(SELECTED)

    with patch("app.agents.orchestrator.nodes_job_tailoring.calculate_compatibility_score_v2",
               _fake_tool(lambda _: json.dumps(REPORT_WITH_RUBRIC))), \
         patch("app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content", _fake_tool(select)), \
         patch("app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content",
               _fake_tool(lambda _: json.dumps(REWRITE))):
        config = {"configurable": {"thread_id": "rubric-flow"}}
        result = graph.invoke({"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB}, config=config)
        preview = _interrupt_payload(result)["preview"]
        assert preview["rubric"][0]["text"] == "Python"
        assert preview["knockouts"] == []

        for _ in range(3):  # score, selection, rewrite
            result = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert seen["evidence"] == "[req.0] must_have Python (met): no bullet shows it"
    values = graph.get_state(config).values
    alignment = values["alignment"]
    assert alignment["before"] == pytest.approx(UNEVIDENCED_FACTOR)
    assert alignment["after"] == pytest.approx(1.0)
    assert alignment["newly_shown"] == ["Python"]
    assert values["requirement_rubric_after"][0]["evidence_bullet_ids"] == [BULLET]
    assert any("Requirement alignment: 0.70 -> 1.00 (+0.30)" in m.content for m in values["messages"])
