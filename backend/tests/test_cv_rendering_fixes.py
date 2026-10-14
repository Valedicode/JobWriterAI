"""
Rendering fixes from a review of a real tailored CV:

- the contact line skipped nothing: invisible icon glyphs from PDF extraction
  printed as empty slots ("• •") and the location appeared twice;
- the strategist's headline was also the summary's first sentence, so the
  summary read twice, the first copy cut short;
- skills were one flat "Skills: ..." list under a "Skills" heading, with the
  spoken languages in a separate tiny section;
- a German grade already labelled "Note" got "Grade:" in front of it.

Run with::

    cd backend
    pytest tests/test_cv_rendering_fixes.py -v
"""

from __future__ import annotations

from app.agents.orchestrator.nodes_job_tailoring import _with_skill_groups
from app.agents.strategist_agent import validate_plan
from app.agents.writer_agent import _format_plan
from app.models.schemas import SkillGroup, StrategistPlan
from app.services.resume_adapter import build_harvard_sections, group_skills

BASE = {
    "name": "Alex Example",
    "email": "alex@example.com",
    "skills": ["Python", "Java", "Django", "React", "PyTorch", "LangChain", "Docker", "Gemma"],
    "languages": [{"language": "German", "fluency": "native"}, {"language": "English", "fluency": "C1"}],
    "education": [{"institution": "Example University", "degree": "B.Sc.", "grade": "1.6"}],
    "experience": [], "projects": [], "leadership_activities": [],
}


def _header(cv):
    header, _ = build_harvard_sections(cv)
    return header


def test_contact_line_skips_invisible_values_and_repeats():
    cv = {
        **BASE,
        "location": "Aachen, 52070",
        "phone": "",                       # a phone icon glyph, extracted as the value
        "portfolio_url": "Aachen, 52070",       # the location again
        "linkedin_url": "​ ",          # zero-width space plus an icon
        "github_url": "https://github.com/example",
    }
    assert _header(cv)["contact_parts"] == ["Aachen, 52070", "alex@example.com", "https://github.com/example"]


def test_a_headline_that_opens_the_summary_is_not_printed_twice():
    summary = "Computer science student with web experience in Python and Java. Seeks a working student role."
    assert _header({**BASE, "label": "Computer science student with web experience in Python and Java",
                    "summary": summary})["label"] == ""
    assert _header({**BASE, "label": "Backend engineer", "summary": summary})["label"] == "Backend engineer"


def test_the_rewriter_is_told_not_to_repeat_the_headline_and_to_keep_the_cv_language():
    plan = {"plan": {"positioning": "p", "headline": "Backend engineer"}}
    assert "do NOT repeat it in the summary" in _format_plan(plan, "rewrite")


def test_flat_skills_are_grouped_by_family_with_languages_as_a_line():
    _, sections = build_harvard_sections(BASE)
    assert "languages" not in [s["type"] for s in sections]
    skills = next(s for s in sections if s["type"] == "skills")["entries"]
    assert skills == [
        {"label": "Programming languages", "text": "Python, Java"},
        {"label": "Web & backend", "text": "Django, React"},
        {"label": "ML & data", "text": "PyTorch, LangChain"},
        {"label": "Tools & infrastructure", "text": "Docker"},
        {"label": "Other", "text": "Gemma"},
        {"label": "Languages", "text": "German (native), English (C1)"},
    ]


def test_german_group_labels():
    assert [g["name"] for g in group_skills(["Python", "Docker", "Gemma"], "german")] == \
        ["Programmiersprachen", "Tools & Infrastruktur", "Weitere"]


def test_a_single_group_prints_without_a_skills_label():
    assert group_skills(["Gemma", "FinBERT"]) == [{"name": "", "keywords": ["Gemma", "FinBERT"]}]
    _, sections = build_harvard_sections({**BASE, "skills": ["Gemma"], "languages": []})
    assert next(s for s in sections if s["type"] == "skills")["entries"] == [{"label": "", "text": "Gemma"}]


def test_the_plans_skill_groups_hold_only_cv_skills():
    plan = StrategistPlan(
        positioning="p", cv_language="english", convention="international",
        skill_groups=[
            SkillGroup(label="ML & agents", skills=["pytorch", "Gemma", "Kubernetes", "LangChain"]),
            SkillGroup(label="Languages", skills=["Python", "Java", "Python"]),
        ],
    )
    cleaned, notes = validate_plan(plan, rubric=[], ledger={}, bullet_index={}, project_count=0,
                                   cv_skills=BASE["skills"])
    assert [(g.label, g.skills) for g in cleaned.skill_groups] == [
        ("ML & agents", ["PyTorch", "Gemma", "LangChain"]),       # written as the CV writes them
        ("Languages", ["Python", "Java"]),
        ("Other", ["Django", "React", "Docker"]),                 # left out by the model, not lost
    ]
    assert any("Kubernetes" in n for n in notes)


def test_the_plans_groups_reach_the_renderer():
    groups = [{"label": "ML & agents", "skills": ["PyTorch", "Gemma"]}, {"label": "Other", "skills": ["Docker"]}]
    cv = _with_skill_groups(BASE, groups)
    _, sections = build_harvard_sections(cv)
    entries = next(s for s in sections if s["type"] == "skills")["entries"]
    assert entries[:2] == [{"label": "ML & agents", "text": "PyTorch, Gemma"}, {"label": "Other", "text": "Docker"}]
    assert _with_skill_groups(BASE, []) is BASE


def test_a_german_grade_label_is_not_prefixed_again():
    cv = {**BASE, "education": [{"institution": "Gymnasium", "degree": "Abitur", "grade": "Note: 1,0"}]}
    _, sections = build_harvard_sections(cv, language="german")
    rows = next(s for s in sections if s["type"] == "education")["entries"][0]["rows"]
    assert [r["left"] for r in rows if "1,0" in r["left"]] == ["Note: 1,0"]
