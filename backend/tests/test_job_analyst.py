"""
Job analyst: the JobRequirements fields beyond skills.

Learnable skills, knockout conditions, posting language and country
round-trip through the schema, and old payloads without them still parse.

Run with::

    cd backend
    pytest tests/test_job_analyst.py -v
"""

from __future__ import annotations

import pytest

from app.agents.orchestrator.nodes_job_tailoring import _job_conditions
from app.agents.scoring_agent import _technical_skill_lists
from app.models.schemas import JobRequirements

BASE = {
    "job_title": "Werkstudent Java Entwicklung",
    "job_level": "student",
    "required_skills": ["Java"],
    "employment_type": "Working student",
    "location": "München",
    "responsibilities": ["Build services"],
    "key_requirements": ["Java"],
}


def test_new_fields_round_trip():
    job = JobRequirements(**BASE, learnable_skills=["Quarkus"], work_mode="hybrid",
                          location_constraint="within commuting distance of Munich",
                          hours_per_week="16-20", start_date="ab sofort",
                          work_permit="EU work permit required", posting_language="de", country="DE")
    again = JobRequirements.model_validate_json(job.model_dump_json())
    assert again == job
    assert again.learnable_skills == ["Quarkus"]


def test_payloads_without_the_new_fields_still_parse():
    job = JobRequirements(**BASE)
    assert job.learnable_skills == []
    assert job.work_mode is None and job.country is None and job.posting_language is None


@pytest.mark.parametrize("raw, expected", [
    ("On-site", "onsite"), ("vor Ort", "onsite"), ("Hybrid (2 days office)", "hybrid"),
    ("Remote-first", "remote"), ("Homeoffice", "remote"), ("flexible", None), (None, None),
])
def test_work_mode_is_normalised_instead_of_failing(raw, expected):
    assert JobRequirements(**BASE, work_mode=raw).work_mode == expected


def test_language_and_country_codes_are_normalised():
    job = JobRequirements(**BASE, posting_language=" DE ", country="de")
    assert job.posting_language == "de" and job.country == "DE"
    assert JobRequirements(**BASE, posting_language="", country="  ").country is None


def test_learnable_skills_still_score_as_nice_to_haves():
    required, preferred = _technical_skill_lists(
        {"required_skills": ["Java"], "preferred_skills": ["Docker"], "learnable_skills": ["Quarkus", "Docker"]}
    )
    assert required == ["Java"]
    assert preferred == ["Docker", "Quarkus"]


def test_job_summary_conditions_list_only_stated_ones():
    assert _job_conditions({"work_mode": "hybrid", "hours_per_week": "16-20", "start_date": None,
                            "work_permit": "  "}) == ["Work mode: hybrid", "Hours/week: 16-20"]
