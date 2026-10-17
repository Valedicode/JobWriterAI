"""Unit tests for LaTeX rendering and JSON Resume adapter."""

from __future__ import annotations

import pytest

from app.services.latex_renderer import latex_escape, latex_url, render_cover_letter_tex, render_cv_tex
from app.services.resume_adapter import build_harvard_sections, to_json_resume


def test_latex_escape_special_characters():
    assert latex_escape("50%") == r"50\%"
    assert latex_escape("R&D") == r"R\&D"
    assert latex_escape("C++") == r"C++"
    assert latex_escape("foo_bar") == r"foo\_bar"


def test_legacy_resume_info_maps_to_json_resume_work():
    cv = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+1 555 0100",
        "skills": ["Python", "Algorithms"],
        "education": ["B.Sc. Mathematics, Cambridge"],
        "experience": [
            {
                "position": "Engineer",
                "company": "Analytical Engines Ltd.",
                "duration": "1843 -- 1852",
                "responsibilities": ["Designed the first algorithm"],
            }
        ],
        "projects": [],
        "leadership_activities": ["Founded the Bayes Reading Group"],
    }
    doc = to_json_resume(cv)
    assert doc["basics"]["name"] == "Ada Lovelace"
    assert doc["work"][0]["name"] == "Analytical Engines Ltd."
    assert doc["work"][0]["highlights"] == ["Designed the first algorithm"]
    assert doc["volunteer"][0]["position"] == "Founded the Bayes Reading Group"


def test_render_cv_tex_includes_harvard_sections():
    cv = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+1 555 0100",
        "skills": ["Python"],
        "education": ["B.Sc. Mathematics"],
        "experience": [
            {
                "position": "Engineer",
                "company": "Analytical Engines Ltd.",
                "duration": "1843 -- 1852",
                "responsibilities": ["Designed the first algorithm"],
            }
        ],
        "projects": [],
        "leadership_activities": [],
    }
    tex = render_cv_tex(cv, section_order=["experience", "education", "skills"])
    assert "Ada Lovelace" in tex
    assert "Analytical Engines Ltd." in tex
    assert r"Designed the first algorithm" in tex or "Designed the first algorithm" in tex
    assert "\\documentclass" in tex


def test_build_harvard_sections_respects_section_order():
    cv = {
        "name": "Test User",
        "email": "t@example.com",
        "phone": "123",
        "skills": ["Go"],
        "education": ["M.S. CS"],
        "experience": [
            {
                "position": "Dev",
                "company": "Co",
                "duration": "2020",
                "responsibilities": ["Built APIs"],
            }
        ],
        "projects": [],
        "leadership_activities": [],
    }
    _, sections = build_harvard_sections(cv, section_order=["skills", "experience", "education"])
    assert [s["type"] for s in sections] == ["skills", "work", "education"]


def test_render_cover_letter_tex_uses_entry_level_header_layout():
    content = {
        "language": "english",
        "opening_paragraph": "I am excited to apply.",
        "body_paragraph_1": "My experience aligns well.",
        "body_paragraph_2": "I bring strong ML skills.",
        "closing_paragraph": "Thank you for your consideration.",
    }
    cv = {
        "name": "Kevin Ha",
        "email": "kevin@example.com",
        "phone": "+49 123 456",
        "linkedin_url": "https://linkedin.com/in/kevinha",
        "github_url": "https://github.com/Valedicode",
    }
    job = {"job_title": "ML Engineer"}
    tex = render_cover_letter_tex(
        content,
        applicant_name="Kevin Ha",
        recipient_info="Dr. Jane Smith",
        cv_data=cv,
        job_data=job,
    )
    assert "\\documentclass[11pt,a4paper]{article}" in tex
    assert "COVER LETTER" in tex
    assert "Kevin Ha" in tex
    assert "ML Engineer" in tex
    assert "kevin@example.com" in tex
    assert "linkedin.com/in/kevinha" in tex
    assert "Yours sincerely" in tex  # a named recipient
    assert "Dear Dr. Jane Smith" in tex


def test_render_cover_letter_tex_requires_recipient():
    content = {
        "language": "english",
        "opening_paragraph": "I am excited to apply.",
        "body_paragraph_1": "My experience aligns well.",
        "body_paragraph_2": "I bring strong ML skills.",
        "closing_paragraph": "Thank you for your consideration.",
    }
    with pytest.raises(ValueError, match="recipient_info is required"):
        render_cover_letter_tex(content, applicant_name="Kevin Ha", recipient_info="")


def test_render_cover_letter_tex_german_includes_betreff():
    content = {
        "language": "german",
        "betreff": "Bewerbung als Softwareentwickler",
        "opening_paragraph": "mit großem Interesse bewerbe ich mich.",
        "body_paragraph_1": "Erster Absatz.",
        "body_paragraph_2": "Zweiter Absatz.",
        "closing_paragraph": "Ich freue mich auf Ihre Rückmeldung.",
        "grussformel": "Mit freundlichen Grüßen",
    }
    tex = render_cover_letter_tex(
        content,
        applicant_name="Max Mustermann",
        recipient_info="Frau Müller",
        cv_data={"name": "Max Mustermann", "email": "max@example.com"},
        job_data={"job_title": "Softwareentwickler"},
    )
    assert "ANSCHREIBEN" in tex
    assert "Betreff:" in tex
    assert "Bewerbung als Softwareentwickler" in tex
    assert "Mit freundlichen Grüßen" in tex


_LETTER = {
    "language": "english",
    "opening_paragraph": "I am excited to apply.",
    "body_paragraph_1": "My experience aligns well.",
    "body_paragraph_2": "I bring strong ML skills.",
    "closing_paragraph": "Thank you for your consideration.",
}


def test_latex_url_keeps_ordinary_urls_working():
    assert latex_url("https://github.com/Valedicode") == "https://github.com/Valedicode"
    assert latex_url("https://x.com/a_b?x=1&y=2#top") == r"https://x.com/a\_b?x=1&y=2\#top"
    assert latex_url("https://linkedin.com/in/jürgen") == r"https://linkedin.com/in/j\%C3\%BCrgen"


@pytest.mark.parametrize("value", [
    "a}{x}\input{/etc/passwd}\textbf{",
    "https://a.com/}{x}^^5cinput{/etc/passwd}^^5ctextbf{",
])
def test_latex_url_cannot_end_the_argument_or_start_a_command(value):
    escaped = latex_url(value)
    assert not any(ch in escaped for ch in "{}^ ")
    assert "\input" not in escaped and "\textbf" not in escaped


@pytest.mark.parametrize("language", ["english", "german"])
def test_cover_letter_email_and_links_cannot_inject_latex(language):
    """cv_data comes from the client: its email and profile URLs must not be
    able to run TeX (\input would read server files into the letter)."""
    tex = render_cover_letter_tex(
        {**_LETTER, "language": language},
        applicant_name="Eve",
        recipient_info="Hiring Team",
        cv_data={
            "name": "Eve",
            "email": "a}{\input{/etc/passwd}}\textbf{",
            "github_url": "https://a.com/}{x}^^5cinput{/etc/passwd}^^5ctextbf{",
        },
    )
    assert "\input" not in tex
    assert "^^5c" not in tex


@pytest.mark.parametrize("compiler", ["tectonic", "pdflatex"])
def test_compile_errors_keep_the_compiler_output_in_the_server_log(compiler, tmp_path, monkeypatch, caplog):
    """The error text reaches the user; the compiler's output (which quotes the
    document and server paths) must only go to the log."""
    import subprocess
    from app.services import latex_renderer

    monkeypatch.setenv("LATEX_COMPILER", compiler)
    (tmp_path / "doc.log").write_text("! Undefined control sequence. l.1 SECRET_LINE", encoding="utf-8")
    monkeypatch.setattr(
        latex_renderer.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="SECRET_LINE from the compiler"),
    )
    with caplog.at_level("ERROR", logger="app.services.latex_renderer"):
        with pytest.raises(RuntimeError) as raised:
            latex_renderer.compile_tex_to_pdf("\documentclass{article}", tmp_path, "doc")
    assert "SECRET_LINE" not in str(raised.value)
    assert "server log" in str(raised.value)
    assert "SECRET_LINE" in caplog.text
