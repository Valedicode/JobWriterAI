"""
Synthetic CVs that exercise every path of the CV adapter (no real person's data).

Used to pin the LaTeX the adapter produces (tests/golden/cv_tex.json) and to check the
editable view built from the same code: every printed string must be exactly the
segments the view shows, and every editable segment must point at the field it came from.
"""

from __future__ import annotations

from typing import Any

from tests.cv_layout_fixtures import cv_with

RICH_EN: dict[str, Any] = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": "+44 20 7946 0000",
    "location": "London, UK",
    "github_url": "https://github.com/ada/",
    "linkedin_url": "https://linkedin.com/in/ada",
    "portfolio_url": "https://ada.dev",
    "label": "Algorithm engineer",
    "summary": "Mathematician and engineer focused on algorithms and their notes.",
    "experience": [
        {"company": "Analytical Engines Ltd.", "position": "Engineer", "location": "London",
         "duration": "January 2020 - Present",
         "responsibilities": ["Designed the first algorithm for the engine.", "  Wrote notes on Bernoulli numbers  ", "", "Shipped 3 releases"]},
        {"company": "", "position": "Private Tutor – Mathematics", "location": "Remote", "duration": "",
         "responsibilities": ["Taught calculus to 12 students."]},
        {"company": "Private Tutor - Mathematics", "position": "Private Tutor – Mathematics",
         "duration": "Mar 2017 – Dec 2017", "responsibilities": []},
    ],
    "education": [
        {"institution": "University of London", "degree": "M.Sc. Mathematics", "location": "London",
         "dates": "October 2019 – Expected 2021", "grade": "1.6 (German System)",
         "details": ["Current Grade: 1.6", "Thesis on difference engines", "Grade: 1.3 in the thesis"]},
        {"institution": "University of London", "degree": "B.Sc. Mathematics", "location": "London",
         "dates": "Sep 2016 -- Jun 2019", "grade": "GPA 3.8", "details": []},
        {"institution": "Gymnasium am Park", "degree": " Abitur ", "location": "Bonn", "dates": "July 2016",
         "grade": "", "details": ["Award for mathematics", "   "]},
    ],
    "leadership_activities": [
        {"organization": "Aachen Investment Club", "role": "Project Manager – Sentiment Analysis", "location": "Aachen",
         "dates": "October 2025 – January 2026", "description": "Led a 5-person student team.",
         "highlights": ["Owned planning and weekly reviews."]},
        {"organization": "", "role": "Founder of the Bayes Reading Group", "location": "", "dates": "",
         "description": "", "highlights": []},
        {"organization": "Chess Club", "role": "Chess Club", "location": "", "dates": "2015",
         "description": "", "highlights": ["Organised a city tournament"]},
    ],
    "projects": [
        {"name": "Difference Engine", "description": "A simulator of the engine",
         "technologies": ["Python", " NumPy ", ""], "outcomes": ["Ran 1,000 tables", "Published results!", ""],
         "url": "https://github.com/ada/engine"},
        {"name": "Notes", "description": "", "technologies": ["Python"],
         "outcomes": ["Technologies: Python", "Explained loops"], "url": ""},
        {"name": "Plain", "description": "Only a description.", "technologies": [], "outcomes": [], "url": ""},
    ],
    "skills": ["Python", "C", "Docker", "PostgreSQL", "LangChain", "Mathematics"],
    "languages": [{"language": "English", "fluency": "native"}, {"language": "German", "fluency": "C1"},
                  {"language": "French", "fluency": None}],
}

# Grouped skills as the strategist's plan writes them, and no label (a summary hides it anyway).
RICH_GROUPED: dict[str, Any] = {
    **RICH_EN,
    "label": None,
    "skills": [{"name": "Languages", "keywords": ["Python", "C"]},
               {"name": "Tools", "keywords": ["Docker", " Git "], "level": "Advanced"},
               {"name": "", "keywords": ["Mathematics"]}],
}

# No summary: the label prints. No skills: languages keep their own section.
NO_SUMMARY_NO_SKILLS: dict[str, Any] = {
    **RICH_EN,
    "summary": "",
    "skills": [],
    "phone": "",  # an icon glyph from a CV template: prints as nothing, so it's dropped
}

# A German CV: German section titles, "Note:" for grades, "Technologien:" in projects.
RICH_DE: dict[str, Any] = {
    "name": "Max Mustermann",
    "email": "max@example.de",
    "phone": "+49 30 000000",
    "location": "Berlin",
    "summary": "Softwareentwickler mit Erfahrung in der Entwicklung von Backend-Systemen und der Arbeit im Team.",
    "experience": [
        {"company": "Beispiel GmbH", "position": "Werkstudent Softwareentwicklung", "location": "Berlin",
         "duration": "April 2024 – heute",
         "responsibilities": ["Entwickelte eine Webanwendung mit Django und PostgreSQL für die Verwaltung.",
                              "Betreute die Nutzer und schrieb die Dokumentation für das Team."]},
    ],
    "education": [
        {"institution": "RWTH Aachen", "degree": "B.Sc. Informatik", "location": "Aachen",
         "dates": "Oktober 2021 – September 2025", "grade": "2,0", "details": ["Grade: 1,7 in der Bachelorarbeit"]},
    ],
    "leadership_activities": [],
    "projects": [{"name": "Analysetool", "description": "Ein Werkzeug für die Analyse von Daten und Berichten.",
                  "technologies": ["Python", "FastAPI"], "outcomes": ["Senkte die Laufzeit um 40 %"],
                  "url": ""}],
    "skills": ["Python", "Django", "FastAPI", "Git"],
    "languages": [{"language": "Deutsch", "fluency": "Muttersprache"}, {"language": "Englisch", "fluency": "C1"}],
}

# The older flat-string shapes for education, activities, projects and experience.
LEGACY_STRINGS: dict[str, Any] = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": "1",
    "skills": ["Python", "Mathematics"],
    "education": ["B.Sc. Computer Science, University of London, 2020", "  M.Sc. Applied Math, in progress "],
    "experience": ["Engineer at Analytical Engines"],
    "leadership_activities": ["Founded the Bayes Reading Group"],
    "projects": ["Difference Engine v0 - Python simulator"],
}

CASES: dict[str, dict[str, Any]] = {
    "rich_en": RICH_EN,
    "rich_grouped": RICH_GROUPED,
    "no_summary_no_skills": NO_SUMMARY_NO_SKILLS,
    "rich_de": RICH_DE,
    "legacy_strings": LEGACY_STRINGS,
    "layout_4x6": cv_with(4, 6),
    "layout_2x3_no_languages": cv_with(2, 3, projects=1, languages=False),
}

# Render options each case is pinned under: (name, kwargs for render_cv_tex).
OPTIONS: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("dach", {"convention": "dach"}),
    ("unmerged", {"merge_languages": False}),
    ("ordered", {"section_order": ["experience", "projects", "education", "skills", "leadership"]}),
    ("compact_break", {"layout": "compact", "break_before_block": 2}),
]
