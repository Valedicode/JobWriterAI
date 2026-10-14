"""
Fact ledger: what the CV and the user state, built without an LLM.

- every bullet, skill, language, date range, employer and title on the CV
  becomes a fact, and the numbers inside bullets do too;
- facts name the technologies they mention, in skill-graph form;
- the run starts with the ledger and a frozen copy of the CV, eligibility
  answers and chat profile additions join it as user-confirmed facts, and
  step 6 no longer overwrites the user's CV with the tailored one.

Run with::

    cd backend
    pytest tests/test_fact_ledger.py -v
"""

from __future__ import annotations

import json

import pytest
from langgraph.types import Command

from app.agents.orchestrator.nodes_job_tailoring import _index_cv_bullets
from app.models.schemas import FactEntry
from app.services.fact_ledger import (
    build_fact_ledger,
    eligibility_answer_facts,
    entities_in,
    metrics_in,
    user_added_fact,
    with_facts,
)

from tests.test_orchestrator_job_tailoring import (  # noqa: F401  (fixtures)
    CV,
    GERMAN_JOB,
    JOB,
    _interrupt_payload,
    graph,
    patched_tools,
    silent_on_german,
)

FULL_CV = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": "+44 1",
    "summary": "Backend engineer working with Python and Kubernetes.",
    "skills": ["Python", "k8s", "SQL"],
    "languages": [{"language": "English", "fluency": "native"}],
    "education": [{
        "institution": "RWTH Aachen", "degree": "M.Sc. Computer Science",
        "dates": "Oct 2023 -- Sep 2025", "grade": "1.6", "details": ["Thesis on query planning"],
    }],
    "experience": [{
        "position": "Software Engineer", "company": "Analytical Engines Ltd.", "duration": "Jan 2022 - Aug 2024",
        "responsibilities": [
            "Cut p95 latency by 30% for 1,200 daily users with FastAPI caching",
            "Mentored 3 interns",
        ],
    }],
    "projects": [{
        "name": "Query Lab", "description": "A SQL playground in React.",
        "technologies": ["React", "PostgreSQL", "Python"], "outcomes": ["Used by 2 courses"],
    }],
    "leadership_activities": [{
        "role": "Tutor", "organization": "Informatik Fachschaft", "dates": "2024",
        "highlights": ["Ran weekly Java exercise sessions"],
    }],
}


@pytest.fixture
def ledger():
    return build_fact_ledger(FULL_CV, _index_cv_bullets(FULL_CV))


def test_every_ledger_entry_is_a_valid_fact(ledger):
    for fact_id, fact in ledger.items():
        assert FactEntry.model_validate(fact).fact_id == fact_id


def test_ledger_covers_bullets_skills_dates_employers_and_titles(ledger):
    bullet_ids = set(_index_cv_bullets(FULL_CV))
    assert bullet_ids <= set(ledger)
    expected = {
        "skill.python", "skill.kubernetes", "skill.sql", "skill.react", "skill.postgresql",
        "lang.english",
        "education.0", "education.0.dates", "education.0.grade", "education.0.details.0",
        "experience.0.title", "experience.0.employer", "experience.0.dates",
        "leadership_activities.0.title", "leadership_activities.0.employer", "leadership_activities.0.dates",
        "projects.0.title",
    }
    assert expected <= set(ledger)
    kinds = {f["kind"] for f in ledger.values()}
    assert {"bullet", "skill", "language", "education", "employer", "title", "date_range", "metric"} <= kinds
    assert all(f["source"] == "cv_extraction" and not f["user_verified"] for f in ledger.values())


def test_bullet_facts_keep_their_text_and_name_their_technologies(ledger):
    bullet = ledger["experience.0.responsibilities.0"]
    assert bullet["kind"] == "bullet" and bullet["bullet_id"] == "experience.0.responsibilities.0"
    assert bullet["value"] == FULL_CV["experience"][0]["responsibilities"][0]
    assert bullet["entities"] == ["fastapi"]
    assert ledger["summary"]["entities"] == ["python", "kubernetes"]
    assert ledger["leadership_activities.0.highlights.0"]["entities"] == ["java"]


def test_numbers_in_bullets_are_facts(ledger):
    metrics = {f["value"] for f in ledger.values()
               if f["kind"] == "metric" and f.get("bullet_id") == "experience.0.responsibilities.0"}
    assert metrics == {"30%", "1,200"}
    assert ledger["experience.0.responsibilities.1.metric.0"]["value"] == "3"


def test_skill_facts_are_normalised_and_deduplicated(ledger):
    assert ledger["skill.kubernetes"]["value"] == "k8s"
    assert ledger["skill.kubernetes"]["entities"] == ["kubernetes"]
    # Python is in the skills list and a project's technologies: one fact, first source kept.
    assert ledger["skill.python"]["source_text"] == "Python"
    assert ledger["skill.react"]["source_text"] == "Project Query Lab: React"


def test_dates_and_grades_carry_no_entities(ledger):
    assert ledger["experience.0.dates"]["entities"] == []
    assert ledger["education.0.grade"]["entities"] == []


def test_education_written_as_plain_strings(ledger):
    plain = build_fact_ledger(CV, _index_cv_bullets(CV))
    assert plain["education.0"]["value"] == "B.Sc. Computer Science"


@pytest.mark.parametrize("text, expected", [
    ("Deployed Go services on k8s with Docker", ["go", "kubernetes", "docker"]),
    ("Let's go to the meeting", []),
    ("Wrote C# and C++ alongside Node.js", ["c#", "c++", "nodejs"]),
    ("Migrated from PG to Postgres 16", ["postgresql"]),
])
def test_entities_in_text(text, expected):
    assert entities_in(text) == expected


def test_years_are_not_metrics():
    assert metrics_in("Shipped in 2023, cut costs by 15% and saved €20k") == ["15%", "€20k"]


def test_eligibility_answers_become_confirmed_facts():
    questions = [
        {"id": "q0", "kind": "language", "language": "German", "requirement": "German (C1)",
         "prompt": "What is your German level?"},
        {"id": "q1", "kind": "language", "language": "French", "requirement": "French (B2)",
         "prompt": "What is your French level?"},
        {"id": "q2", "kind": "education", "requirement": "Enrolled student in Computer Science",
         "prompt": "Are you enrolled?"},
    ]
    facts = {f["fact_id"]: f for f in eligibility_answer_facts(
        questions, {"q0": "C1", "q1": "none", "q2": "Expected: September 2027"},
    )}
    assert facts["lang.german"]["value"] == "German (C1)"
    assert facts["user.french"]["kind"] == "user_stated"
    assert facts["user.french"]["value"] == "Does not speak French"
    education = facts["user.enrolled-student-in-computer-science"]
    assert education["value"] == "Expected: September 2027"
    assert all(f["source"] == "user_confirmed" and f["user_verified"] for f in facts.values())


def test_user_added_facts_replace_by_id():
    ledger = with_facts({}, [user_added_fact("skill", "Docker", None, "I used Docker")])
    ledger = with_facts(ledger, [user_added_fact("language", "German", "C1", "I speak German")])
    assert ledger["skill.docker"]["entities"] == ["docker"]
    assert ledger["lang.german"]["value"] == "German (C1)"
    assert with_facts(ledger, [user_added_fact("language", "German", "C2", "x")])["lang.german"]["value"] == "German (C2)"


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

def test_the_run_starts_with_the_ledger_and_a_frozen_cv(graph, patched_tools):  # noqa: F811
    config = {"configurable": {"thread_id": "ledger-start"}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    values = graph.get_state(config).values
    assert values["original_cv_data"] == CV
    assert set(_index_cv_bullets(CV)) <= set(values["fact_ledger"])
    assert "skill.python" in values["fact_ledger"]


def test_step6_keeps_the_users_cv_and_writes_the_tailored_one(graph, patched_tools):  # noqa: F811
    config = {"configurable": {"thread_id": "ledger-step6"}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    for _ in range(3):  # score, selection, rewrite
        graph.invoke(Command(resume={"action": "approve"}), config=config)
    values = graph.get_state(config).values
    assert values["cv_data"] == CV
    assert values["original_cv_data"] == CV
    assert values["tailored_cv_data"]["experience"][0]["responsibilities"][0] != \
        CV["experience"][0]["responsibilities"][0]


def test_eligibility_answers_join_the_ledger(graph, silent_on_german):  # noqa: F811
    config = {"configurable": {"thread_id": "ledger-eligibility"}}
    result = graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": GERMAN_JOB}, config=config)
    [question] = _interrupt_payload(result)["preview"]["questions"]
    graph.invoke(Command(resume={"action": "edit", "feedback": "C1", "answers": {question["id"]: "C1"}}),
                 config=config)
    fact = graph.get_state(config).values["fact_ledger"]["lang.german"]
    assert fact["value"] == "German (C1)"
    assert fact["source"] == "user_confirmed" and fact["user_verified"] is True
    assert json.dumps(fact)  # stays plain JSON for the checkpointer
