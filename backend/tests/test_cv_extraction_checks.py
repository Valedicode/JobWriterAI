"""
Extraction slips the PDF text contradicts are removed after the model's extraction.

The first case reproduces a real run: an activity with only a paragraph came back with the
three bullets of the jobs listed after it, and a tutoring job got its title as its company.

Run with::

    cd backend
    pytest tests/test_cv_extraction_checks.py -v
"""

from __future__ import annotations

import copy

from app.services.cv_extraction_checks import (
    clean_extracted_cv,
    clear_company_echoes,
    drop_cross_entry_duplicates,
    drop_grade_echoes,
)

YAMS = ("Developed YAMS, a Python/Django/PostgreSQL management web application for institute inquiries "
        "and acquisition data, adopted by institute staff.")
TUTOR = ("Designed personalised lesson plans for high school mathematics students, supporting them in "
         "improving by at least one grade level.")
THEATRE = "Managed audience services and customer inquiries during live theatre performances."

# The PDF's text as PyMuPDF reads it: left and right columns on separate lines,
# bullets wrapped over lines, one hyphenated line break.
SOURCE = f"""Leadership & Activities
Aachen Investment Club
Aachen, Germany
Project Manager – Sentiment Analysis Initiative
October 2025 – January 2026
Led a 5-person student team building a financial sentiment analysis system (see Financial Senti-
ment Analyzer); owned planning, task allocation, and weekly reviews.
Experience
Fraunhofer-Institut für Produktionstechnologie IPT
Aachen, Germany
Internship – Software Development
April 2024 – July 2024
• Developed YAMS, a Python/Django/PostgreSQL management web application for institute inquiries
and acquisition data, adopted by institute staff.
Private Tutor – Mathematics
April 2022 – December 2024
• Designed personalised lesson plans for high school mathematics students, supporting them in
improving by at least one grade level.
Theater Krefeld und Mönchengladbach gGmbH
Theatre Service Staff
October 2022 – August 2025
• {THEATRE}
"""


def _extracted() -> dict:
    """What the model returned in the bad run."""
    return {
        "name": "Ada",
        "leadership_activities": [{
            "organization": "Aachen Investment Club",
            "role": "Project Manager – Sentiment Analysis Initiative",
            "description": "Led a 5-person student team building a financial sentiment analysis system.",
            "highlights": [YAMS, TUTOR, THEATRE],
        }],
        "experience": [
            {"company": "Fraunhofer-Institut für Produktionstechnologie IPT",
             "position": "Internship – Software Development", "responsibilities": [YAMS]},
            {"company": "Private Tutor – Mathematics", "position": "Private Tutor – Mathematics",
             "responsibilities": [TUTOR]},
            {"company": "Theater Krefeld und Mönchengladbach gGmbH", "position": "Theatre Service Staff",
             "responsibilities": [THEATRE]},
        ],
        "education": [{"institution": "RWTH Aachen University", "degree": "M.Sc. in Computer Science",
                       "grade": "1.6 (German System)", "details": ["Current Grade: 1.6", "Thesis on agents"]}],
    }


def test_bullets_copied_into_an_activity_are_removed_and_stay_with_their_jobs():
    cv = _extracted()
    notes = drop_cross_entry_duplicates(cv, SOURCE)
    assert cv["leadership_activities"][0]["highlights"] == []
    assert [e["responsibilities"] for e in cv["experience"]] == [[YAMS], [TUTOR], [THEATRE]]
    assert len(notes) == 3


def test_a_slightly_reworded_copy_is_still_recognised():
    cv = _extracted()
    cv["leadership_activities"][0]["highlights"] = [YAMS.replace("web application", "website")]
    drop_cross_entry_duplicates(cv, SOURCE)
    assert cv["leadership_activities"][0]["highlights"] == []
    assert cv["experience"][0]["responsibilities"] == [YAMS]


def test_a_bullet_the_pdf_itself_repeats_is_left_alone():
    cv = _extracted()
    notes = drop_cross_entry_duplicates(cv, SOURCE + f"\nAnother role\n• {YAMS}\n")
    assert YAMS in cv["leadership_activities"][0]["highlights"]
    assert any("found 2 times" in n for n in notes)


def test_a_bullet_missing_from_the_pdf_text_is_left_alone():
    cv = _extracted()
    drop_cross_entry_duplicates(cv, "Unrelated text with no bullets at all.")
    assert cv["leadership_activities"][0]["highlights"] == [YAMS, TUTOR, THEATRE]


def test_a_project_named_inside_another_entrys_text_is_not_taken_for_its_header():
    # "Financial Sentiment Analyzer" appears inside the activity's paragraph; only a
    # line that starts with an entry's name counts as that entry's header.
    cv = {
        "leadership_activities": [{"organization": "Aachen Investment Club", "role": "Project Manager",
                                   "highlights": ["Built the sentiment pipeline for the club newsletter."]}],
        "projects": [{"name": "Financial Sentiment Analyzer",
                      "outcomes": ["Built the sentiment pipeline for the club newsletter."]}],
    }
    source = ("Aachen Investment Club\nProject Manager\nLed the team behind the Financial Sentiment Analyzer.\n"
              "• Built the sentiment pipeline for the club newsletter.\n"
              "Projects\nFinancial Sentiment Analyzer\nA FinBERT pipeline.\n")
    drop_cross_entry_duplicates(cv, source)
    assert cv["leadership_activities"][0]["highlights"] == ["Built the sentiment pipeline for the club newsletter."]
    assert cv["projects"][0]["outcomes"] == []


def test_a_detail_repeating_the_grade_is_dropped():
    cv = _extracted()
    notes = drop_grade_echoes(cv)
    assert cv["education"][0]["details"] == ["Thesis on agents"]
    assert len(notes) == 1


def test_only_details_restating_the_same_grade_are_dropped():
    cv = {"education": [{"grade": "1.6", "details": ["Thesis grade: 1.3", "GPA 1.6 in the major"]}]}
    drop_grade_echoes(cv)
    assert cv["education"][0]["details"] == ["Thesis grade: 1.3"]


def test_a_company_repeating_the_position_is_cleared():
    cv = _extracted()
    notes = clear_company_echoes(cv)
    assert cv["experience"][1]["company"] is None
    assert cv["experience"][0]["company"].startswith("Fraunhofer")
    assert len(notes) == 1


def test_clean_extracted_cv_runs_every_check_without_touching_its_input():
    cv = _extracted()
    before = copy.deepcopy(cv)
    cleaned, notes = clean_extracted_cv(cv, SOURCE)
    assert cv == before
    assert cleaned["leadership_activities"][0]["highlights"] == []
    assert cleaned["education"][0]["details"] == ["Thesis on agents"]
    assert cleaned["experience"][1]["company"] is None
    assert len(notes) == 5


def test_the_cv_upload_tool_returns_the_checked_extraction():
    """extract_resume_info runs the checks on what the model returned, against the PDF's text."""
    from unittest.mock import patch

    import pymupdf
    from langchain_core.runnables import RunnableLambda

    from app.agents import cv_agent
    from app.models.schemas import ResumeInfo

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(36, 36, 560, 800), SOURCE, fontsize=8)
    pdf_bytes = doc.tobytes()

    extracted = ResumeInfo(**{**_extracted(), "email": "ada@example.com", "phone": "1", "skills": [], "projects": []})

    class _Model:
        def __call__(self, role, *, temperature=None):
            return self

        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: extracted)

    with patch.object(cv_agent, "chat_model", _Model()):
        out = ResumeInfo.model_validate_json(cv_agent.extract_resume_info.invoke({"pdf_bytes": pdf_bytes}))

    assert out.leadership_activities[0].highlights == []
    assert [e.responsibilities for e in out.experience] == [[YAMS], [TUTOR], [THEATRE]]
    assert out.experience[1].company is None
    assert out.education[0].details == ["Thesis on agents"]


def test_the_cv_prints_a_repeated_title_and_a_repeated_grade_once():
    """Render-time guard for CVs extracted before the checks existed (or sent by a client)."""
    from app.services.latex_renderer import render_cv_tex

    cv = {
        "name": "Ada", "email": "ada@example.com", "phone": "1", "skills": ["Python"], "projects": [],
        "experience": [{"company": "Private Tutor – Mathematics", "position": "Private Tutor - Mathematics",
                        "duration": "April 2022 – December 2024", "responsibilities": [TUTOR]}],
        "education": [{"institution": "RWTH Aachen University", "degree": "M.Sc. in Computer Science",
                       "dates": "October 2025 – 2028", "grade": "1.6", "details": ["Current Grade: 1.6", "Thesis on agents"]}],
    }
    tex = render_cv_tex(cv)
    assert tex.count("Private Tutor") == 1
    assert tex.count("1.6") == 1
    assert "Thesis on agents" in tex
