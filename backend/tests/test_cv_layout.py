"""
CV formatter: page-break rules, CV conventions, salutations, parse-back.

Page-break rules (user-reported):
- no orphan section on the last page, and no nearly empty last page:
  compact re-render, then Languages folded into Skills, then whole blocks
  moved to page 2; flagged when none of it works;
- a heading keeps with its first entries; entries never split; no section
  leaves a single entry on one side of a break; no widow or club lines.

The compile tests render real PDFs and are skipped without a LaTeX compiler.

Run with::

    cd backend
    pytest tests/test_cv_layout.py -v
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata

import pytest

from app.agents.orchestrator.nodes_job_tailoring import _with_project_order
from app.services.bullet_index import index_cv_bullets
from app.services.latex_renderer import (
    _cover_letter_salutation,
    _english_sign_off,
    _first_block_on_page_two,
    _page_blocks,
    cv_content_language,
    layout_problems,
    render_cv_pdf,
    render_cv_tex,
)
from app.services.resume_adapter import build_harvard_sections, format_dates_dach

from tests.cv_layout_fixtures import cv_with

ORDER = ["experience", "education", "projects", "skills", "languages"]


# --------------------------------------------------------------------------- #
# Blocks: what LaTeX may not break inside
# --------------------------------------------------------------------------- #

def _entries(n: int, bullets: int = 2) -> list[dict]:
    return [{"rows": [{"left": f"Company {i}", "right": "", "bold": True}], "bullets": ["x"] * bullets}
            for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_sections_move_as_a_whole(n):
    blocks = _page_blocks({"style": "entries", "entries": _entries(n)})
    assert len(blocks) == 1 and blocks[0]["heading"] and len(blocks[0]["entries"]) == n


@pytest.mark.parametrize("n", [4, 5, 7])
def test_long_sections_break_with_two_entries_on_each_side(n):
    blocks = _page_blocks({"style": "entries", "entries": _entries(n)})
    assert blocks[0]["heading"] and len(blocks[0]["entries"]) == 2  # heading keeps with its first entries
    assert len(blocks[-1]["entries"]) == 2
    assert sum(len(b["entries"]) for b in blocks) == n
    assert not any(b["heading"] for b in blocks[1:])


def test_a_section_too_tall_for_one_page_is_still_breakable():
    blocks = _page_blocks({"style": "entries", "entries": _entries(3, bullets=20)})
    assert len(blocks) == 3 and blocks[0]["heading"]


def test_the_template_forbids_widows_clubs_and_breaks_inside_blocks():
    tex = render_cv_tex(cv_with(2, 2), ORDER)
    assert r"\widowpenalty=10000" in tex and r"\clubpenalty=10000" in tex
    assert tex.count(r"\begin{keeptogether}") == tex.count(r"\end{keeptogether}") > 0


def test_a_block_can_be_moved_to_the_next_page():
    tex = render_cv_tex(cv_with(4, 2), ORDER, break_before_block=1)
    assert tex.count(r"\newpage") == 1


# --------------------------------------------------------------------------- #
# Page checks
# --------------------------------------------------------------------------- #

def _analysis(pages=2, first=0.9, last=0.5, lines=20):
    return {"pages": pages, "first_page_fill": first, "last_page_fill": last, "last_page_lines": lines}


@pytest.mark.parametrize("analysis, expected", [
    (_analysis(pages=1, last=0.1, lines=2), []),
    (_analysis(lines=3), ["the last page holds only 3 line(s)"]),
    (_analysis(last=0.2), ["the last page is only 20% full"]),
    (_analysis(first=0.4), ["the first page is only 40% full"]),
    (_analysis(pages=3), ["the CV runs to 3 pages"]),
    (_analysis(), []),
])
def test_layout_problems(analysis, expected):
    assert layout_problems(analysis) == expected


def test_the_block_starting_page_two_is_found_by_its_text():
    anchors = ["Experience", "Company 3", "Education"]
    assert _first_block_on_page_two("Company 3                        Berlin", anchors) == 1
    assert _first_block_on_page_two("Education", anchors) == 2
    assert _first_block_on_page_two("ﬁller text", anchors) is None


# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, language, expected", [
    ("Jan 2022 -- Aug 2024", "english", "01/2022 – 08/2024"),
    ("March 2021 - Present", "english", "03/2021 – present"),
    ("Oktober 2020 bis heute", "german", "10/2020 – heute"),
    ("Sept. 2019 to Dec 2020", "english", "09/2019 – 12/2020"),
    ("2018 -- 2021", "english", "2018 -- 2021"),
    ("Berlin", "english", "Berlin"),
])
def test_dach_dates(text, language, expected):
    assert format_dates_dach(text, language) == expected


def test_section_titles_follow_the_language_of_the_cv_content():
    german = cv_with(1, 1)
    german["experience"][0]["responsibilities"] = ["Entwicklung der API mit Python und die Betreuung der Kunden"]
    assert cv_content_language(german) == "german"
    assert cv_content_language(cv_with(1, 1)) == "english"
    _, sections = build_harvard_sections(german, ORDER, language="german")
    assert [s["title"] for s in sections] == ["Berufserfahrung", "Ausbildung", "Projekte", "Kenntnisse", "Sprachen"]
    grade_row = sections[1]["entries"][0]["rows"][-1]
    assert grade_row["left"] == "Note: 1.7"
    assert "Technologien:" in sections[2]["entries"][0]["text"]


def test_languages_can_be_folded_into_skills():
    _, sections = build_harvard_sections(cv_with(1, 1), ORDER, merge_languages=True)
    assert "languages" not in [s["type"] for s in sections]
    skills = next(s for s in sections if s["type"] == "skills")
    assert skills["entries"][-1] == {"label": "Languages", "text": "German (native), English (C1)"}


def test_the_plans_project_order_is_applied():
    cv = {"projects": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
    assert [p["name"] for p in _with_project_order(cv, [2, 0])["projects"]] == ["C", "A", "B"]
    assert _with_project_order(cv, []) is cv


@pytest.mark.parametrize("recipient, expected", [
    ("Frau Dr. Weber", "Sehr geehrte Frau Dr. Weber,"),
    ("Herr Schmidt", "Sehr geehrter Herr Schmidt,"),
    ("Alex Weber", "Guten Tag Alex Weber,"),
    ("Recruiting-Team", "Sehr geehrte Damen und Herren,"),
])
def test_german_salutations(recipient, expected):
    assert _cover_letter_salutation(recipient, is_german=True) == expected


def test_english_sign_off_follows_the_recipient():
    assert _english_sign_off("Dr. Jane Smith") == "Yours sincerely"
    assert _english_sign_off("Hiring Team") == "Yours faithfully"


# --------------------------------------------------------------------------- #
# Rendered PDFs (need a LaTeX compiler)
# --------------------------------------------------------------------------- #

@pytest.fixture(params=["pdflatex", "tectonic"])
def compiler(request, monkeypatch):
    """Each installed compiler: pdfLaTeX (8-bit fonts) and Tectonic (XeTeX,
    Unicode fonts) need different font setups, so both are exercised."""
    if request.param == "pdflatex":
        if not shutil.which("pdflatex"):
            pytest.skip("pdflatex not installed")
    elif not (os.getenv("TECTONIC_PATH") or shutil.which("tectonic")):
        pytest.skip("tectonic not installed (set TECTONIC_PATH to test it)")
    monkeypatch.setenv("LATEX_COMPILER", request.param)
    return request.param


def _pdf_lines(path) -> list[list[str]]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [[line for line in page.get_text().splitlines() if line.strip()] for page in doc]


def _flat(text: str) -> str:
    """PDF text without line breaks, hyphenation or ligatures, for comparison."""
    text = unicodedata.normalize("NFKC", text).replace("-\n", "")
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("jobs, bullets", [(3, 5), (3, 6), (4, 5), (4, 6), (4, 7), (5, 6)])
def test_rendered_cvs_keep_the_page_rules(compiler, tmp_path, jobs, bullets):
    cv = cv_with(jobs, bullets)
    pdf, report = render_cv_pdf(cv, tmp_path, "cv", section_order=ORDER)
    assert report["problems"] == []
    pages = _pdf_lines(pdf)
    titles = {"Experience", "Education", "Projects", "Skills", "Languages"}
    for page in pages[:-1]:
        assert page[-1] not in titles, "a heading ended a page"
    if len(pages) > 1:
        # Page 2 starts with a whole block: an entry's title row or a heading.
        assert pages[1][0].startswith(("Company", "Example University")) or pages[1][0] in titles
        assert len(pages[-1]) > 4, "a short section was stranded on the last page"
    leftovers = sorted(p.name for p in tmp_path.iterdir())
    assert leftovers == ["cv.pdf"]


def test_parse_back_recovers_every_bullet(compiler, tmp_path):
    cv = cv_with(4, 6)
    pdf, _ = render_cv_pdf(cv, tmp_path, "cv", section_order=ORDER, convention="dach")
    text = _flat("\n".join("\n".join(page) for page in _pdf_lines(pdf)))
    for bullet_id, ref in index_cv_bullets(cv).items():
        assert _flat(ref["text"]) in text, bullet_id
    for entry in cv["experience"]:
        assert entry["company"] in text and entry["position"] in text
    assert "01/2011 – 12/2012" in text


def test_german_characters_dashes_and_bullets_survive(compiler, tmp_path):
    # Regression: with T1 fonts under Tectonic the dash vanished and the sharp
    # s printed as "SS"; under pdfLaTeX they extracted as control characters.
    cv = cv_with(1, 1)
    cv["experience"][0]["responsibilities"] = ["Straße und Grüße – mit Umlauten äöü"]
    pdf, _ = render_cv_pdf(cv, tmp_path, "cv", section_order=ORDER, convention="dach")
    text = _flat("\n".join("\n".join(page) for page in _pdf_lines(pdf)))
    assert "Straße und Grüße – mit Umlauten äöü" in text
    assert "• Berlin" in text


def test_cover_letters_keep_german_characters_and_dashes(compiler, tmp_path):
    from app.services.latex_renderer import compile_tex_to_pdf, render_cover_letter_tex

    content = {"language": "german", "betreff": "Bewerbung", "opening_paragraph": "Gerne – mit Grüßen.",
               "body_paragraph_1": "Eins.", "body_paragraph_2": "Zwei.", "closing_paragraph": "Ende.",
               "grussformel": "Mit freundlichen Grüßen"}
    tex = render_cover_letter_tex(content, "Alex", recipient_info="Frau Weber",
                                  cv_data={"name": "Alex", "email": "a@example.com"}, job_data={"job_title": "Dev"})
    text = _flat("\n".join("\n".join(page) for page in _pdf_lines(compile_tex_to_pdf(tex, tmp_path, "letter"))))
    assert "Sehr geehrte Frau Weber," in text
    assert "Gerne – mit Grüßen." in text and "Mit freundlichen Grüßen" in text
