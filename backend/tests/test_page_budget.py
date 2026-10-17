"""
A CV takes the fewest pages a readable layout allows, and keeps to a recommended length.

Before: a CV whose normal spacing spilled a few lines onto page 2 stayed on two pages
(the compact layout was only tried when the two-page layout broke a rule), and the
review's "Recommended pages: 1" was shown but never passed to the renderer.

The compiler is faked: each layout reports a page count, so the choice is tested exactly.

Run with::

    cd backend
    pytest tests/test_page_budget.py -v
"""

from __future__ import annotations

import json

import pytest

from app.services import latex_renderer

CV = {"name": "Ada", "email": "ada@example.com", "phone": "1", "skills": ["Python"], "projects": [],
      "education": [], "experience": [{"company": "Analytical Engines", "position": "Engineer",
                                       "responsibilities": ["Designed the first algorithm"]}]}


def _fake_compiler(monkeypatch, pages_by_layout):
    """compile_tex_to_pdf writes a stub PDF; analyze_cv_pdf reports the pages
    configured for the layout the TeX was rendered with."""
    rendered = {}

    def compile_tex_to_pdf(tex, output_dir, job_name):
        layout = "compact" if latex_renderer.CV_LAYOUTS["compact"]["linespread"] in tex else "normal"
        path = output_dir / f"{job_name}.pdf"
        path.write_bytes(b"%PDF-1.4 stub")
        rendered[path.name] = layout
        return path

    def analyze_cv_pdf(path):
        pages = pages_by_layout[rendered[path.name]]
        return {"pages": pages, "first_page_fill": 0.95, "last_page_fill": 0.9 if pages == 1 else 0.6,
                "last_page_lines": 30, "page_two_first_line": ""}

    monkeypatch.setattr(latex_renderer, "compile_tex_to_pdf", compile_tex_to_pdf)
    monkeypatch.setattr(latex_renderer, "analyze_cv_pdf", analyze_cv_pdf)


def test_a_cv_that_fits_one_page_compactly_gets_one_page(tmp_path, monkeypatch):
    _fake_compiler(monkeypatch, {"normal": 2, "compact": 1})
    pdf, report = latex_renderer.render_cv_pdf(CV, tmp_path, "cv")
    assert (report["layout"], report["pages"], report["problems"]) == ("compact", 1, [])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cv.pdf"]


def test_a_cv_that_needs_two_pages_keeps_normal_spacing(tmp_path, monkeypatch):
    _fake_compiler(monkeypatch, {"normal": 2, "compact": 2})
    _, report = latex_renderer.render_cv_pdf(CV, tmp_path, "cv")
    assert (report["layout"], report["pages"], report["problems"]) == ("normal", 2, [])


def test_a_recommended_two_pages_keeps_normal_spacing(tmp_path, monkeypatch):
    _fake_compiler(monkeypatch, {"normal": 2, "compact": 1})
    _, report = latex_renderer.render_cv_pdf(CV, tmp_path, "cv", max_pages=2)
    assert (report["layout"], report["pages"]) == ("normal", 2)


def test_missing_a_recommended_length_is_reported_not_hidden(tmp_path, monkeypatch):
    _fake_compiler(monkeypatch, {"normal": 2, "compact": 2})
    _, report = latex_renderer.render_cv_pdf(CV, tmp_path, "cv", max_pages=1)
    assert report["pages"] == 2
    assert report["problems"] == ["the CV needs 2 pages, more than the recommended 1, even with compact spacing"]


def test_the_cv_review_passes_its_recommended_length_and_reports_layout_notes(monkeypatch):
    from types import SimpleNamespace

    from app.agents.orchestrator import nodes_cv_review

    received = {}

    def _generate(kwargs):
        received.update(kwargs)
        return (f"CV PDF generated successfully! The file '{kwargs['output_filename']}' is ready for download.\n"
                "Layout note: the CV needs 2 pages, more than the recommended 1, even with compact spacing; "
                "consider shortening or removing a section.")

    monkeypatch.setattr(nodes_cv_review, "generate_cv_pdf", SimpleNamespace(invoke=_generate))
    state = {"flow": "cv_review", "cv_data": CV, "review_outputs": {"assessment": {"recommended_pages": 1}}}
    update = nodes_cv_review.cr_step7_finalize(state, {"configurable": {}})
    assert received["max_pages"] == 1
    assert "Layout note: the CV needs 2 pages" in update["last_narration"]


@pytest.mark.parametrize("recommended, expected", [(2, 2), (None, 0), (5, 0)])
def test_only_a_valid_recommendation_is_passed_on(monkeypatch, recommended, expected):
    from types import SimpleNamespace

    from app.agents.orchestrator import nodes_cv_review

    received = {}
    monkeypatch.setattr(nodes_cv_review, "generate_cv_pdf", SimpleNamespace(
        invoke=lambda kw: received.update(kw) or f"The file '{kw['output_filename']}' is ready for download."))
    state = {"flow": "cv_review", "cv_data": CV, "review_outputs": {"assessment": {"recommended_pages": recommended}}}
    nodes_cv_review.cr_step7_finalize(state, {"configurable": {}})
    assert received["max_pages"] == expected
