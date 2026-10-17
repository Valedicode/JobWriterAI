"""
The CV adapter builds every printed string from segments that know their source field.

The CV editor shows the CV as the PDF prints it and writes each edit back to the field
it came from. That only works if (1) the PDF is unchanged by the segments (pinned
against tests/golden/cv_tex.json), (2) every printed string is exactly its segments, and
(3) every editable segment points at a field whose value is what it shows.

Run with::

    cd backend
    pytest tests/test_cv_view.py -v
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from app.services.latex_renderer import cv_content_language
from app.services.resume_adapter import build_harvard_sections, build_harvard_view
from tests.cv_view_fixtures import CASES, OPTIONS, RICH_DE, RICH_EN, RICH_GROUPED, LEGACY_STRINGS
from tests.golden.regenerate_cv_tex import GOLDEN, render_all


def _build(cv, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in ("section_order", "convention", "merge_languages")}
    return build_harvard_view(cv, kwargs.pop("section_order", None), language=cv_content_language(cv), **kwargs)


def _elements(header, sections):
    """(printed string, its segments) for everything the CV prints."""
    yield header["name"], header["name_segs"]
    if header["label"]:
        yield header["label"], header["label_segs"]
    if header["summary"]:
        yield header["summary"], header["summary_segs"]
    yield from zip(header["contact_parts"], header["contact_segs"])
    for section in sections:
        for entry in section["entries"]:
            if "rows" in entry:
                for row in entry["rows"]:
                    yield row["left"], row["left_segs"]
                    yield row["right"], row["right_segs"]
                yield from zip(entry["bullets"], entry["bullet_segs"])
            else:
                yield entry["label"], entry["label_segs"]
                yield entry["text"], entry["text_segs"]


def _resolve(data, path):
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data.get(part)
    return data


def _normal(value) -> str:
    if isinstance(value, list):
        return ", ".join(v.strip() for v in value if isinstance(v, str) and v.strip())
    if isinstance(value, dict):  # a structured location
        value = ", ".join(v for v in value.values() if isinstance(v, str) and v.strip())
    return re.sub(r"\s+", " ", str(value or "")).strip()


def test_the_pdf_is_unchanged():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rendered = render_all()
    assert [k for k in golden if golden[k] != rendered.get(k)] == []


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("option, kwargs", OPTIONS)
def test_every_printed_string_is_its_segments_and_every_segment_names_its_field(case, option, kwargs):
    cv = CASES[case]
    before = copy.deepcopy(cv)
    header, sections = _build(cv, **kwargs)
    for printed, segs in _elements(header, sections):
        assert printed == "".join(s["text"] for s in segs)
        for seg in segs:
            for path in seg["src"]:
                raw = seg.get("value", seg["text"])
                field = _resolve(cv, path)
                if seg.get("kind") == "list":
                    assert _normal(raw.split(",")) == _normal(field), (path, raw, field)
                else:
                    assert _normal(raw) == _normal(field), (path, raw, field)
    assert cv == before, "building the view modified the CV"


def _section(sections, kind):
    return next(s for s in sections if s["type"] == kind)


def test_formatted_and_shared_fields_keep_their_source():
    header, sections = _build(RICH_EN, convention="dach")
    work = _section(sections, "work")["entries"][0]
    dates = work["rows"][1]["right_segs"][0]
    assert dates == {"text": "01/2020 – present", "src": ["experience.0.duration"], "value": "January 2020 - Present"}
    # An empty bullet is skipped, so the next one still points at its own index.
    assert [s["src"] for s in work["bullet_segs"][-1]] == [["experience.0.responsibilities.3"]]
    assert work["bullets_src"] == ["experience.0.responsibilities"]

    london = _section(sections, "education")["entries"][0]["rows"][0]["left_segs"][0]
    assert london["src"] == ["education.0.institution", "education.1.institution"]  # one header for both degrees


def test_the_adapters_own_words_are_not_editable():
    _, sections = _build(RICH_EN)
    grade = _section(sections, "education")["entries"][0]["rows"][2]["left_segs"]
    assert grade[0] == {"text": "Grade: ", "src": []}
    assert grade[1]["src"] == ["education.0.grade"]

    project = _section(sections, "projects")["entries"][0]["text_segs"]
    tech = next(s for s in project if s.get("kind") == "list")
    assert tech["src"] == ["projects.0.technologies"]
    assert {"text": "Technologies: ", "src": []} in project

    skills = _section(sections, "skills")["entries"]
    spoken = skills[-1]
    assert spoken["label_segs"] == [{"text": "Languages", "src": []}]
    assert ["languages.1.fluency"] in [s["src"] for s in spoken["text_segs"]]
    # A flat skills list is grouped by the renderer: its groups have no field to edit.
    assert all(s["src"] == [] for e in skills[:-1] for s in e["label_segs"] + e["text_segs"])


def test_german_wording_is_printed_and_the_raw_value_kept():
    _, sections = _build(RICH_DE, convention="dach")
    rows = _section(sections, "education")["entries"][0]["rows"]
    assert rows[2]["left_segs"] == [{"text": "Note: ", "src": []},
                                    {"text": "2,0", "src": ["education.0.grade"]}]
    detail = rows[3]["left_segs"][0]
    assert detail["text"] == "Note: 1,7 in der Bachelorarbeit"
    assert detail["value"] == "Grade: 1,7 in der Bachelorarbeit"


def test_grouped_skills_are_editable_groups():
    _, sections = _build(RICH_GROUPED)
    tools = _section(sections, "skills")["entries"][1]
    assert tools["label_segs"][0]["src"] == ["skills.1.name"]
    assert tools["text_segs"][0] == {"text": "Docker, Git", "src": ["skills.1.keywords"], "kind": "list"}
    assert tools["text_segs"][2]["src"] == ["skills.1.level"]


def test_flat_string_entries_point_at_the_string_itself():
    header, sections = _build(LEGACY_STRINGS)
    assert _section(sections, "education")["entries"][0]["rows"][0]["left_segs"][0]["src"] == ["education.0"]
    assert _section(sections, "work")["entries"][0]["rows"][0]["left_segs"][0]["src"] == ["experience.0"]
    assert _section(sections, "volunteer")["entries"][0]["rows"][0]["left_segs"][0]["src"] == ["leadership_activities.0"]
    assert _section(sections, "projects")["entries"][0]["label_segs"][0]["src"] == ["projects.0"]
    assert header["name_segs"][0]["src"] == ["name"]


def test_json_resume_input_prints_the_same_but_offers_nothing_to_edit():
    from app.services.resume_adapter import to_json_resume

    doc = {k: v for k, v in to_json_resume(copy.deepcopy(RICH_EN)).items()}
    doc["basics"] = {k: v for k, v in doc["basics"].items() if k != "_keys"}
    for key in ("work", "education", "volunteer", "projects", "skills", "languages"):
        doc[key] = [{k: v for k, v in item.items() if not k.startswith("_src") and k != "_keys"}
                    if isinstance(item, dict) else item for item in doc[key]]
    header, sections = build_harvard_view(doc)
    assert all(not seg["src"] for _, segs in _elements(header, sections) for seg in segs)


def test_the_printed_sections_keep_their_plain_shape():
    """The template's input stays strings only; the segments are the view's."""
    header, sections = build_harvard_sections(RICH_EN)
    assert set(header) == {"name", "label", "summary", "contact_parts"}
    entry = sections[0]["entries"][0]
    assert all(isinstance(v, (str, bool, list)) for v in entry.values())
    assert "bullet_segs" not in entry and "bullets_src" not in entry
