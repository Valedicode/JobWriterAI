"""
Adapt internal CV data to JSON Resume shape for single-column LaTeX rendering.

Reference taxonomy: https://jsonresume.org/schema

Accepts either:
- JSON Resume documents (``basics``, ``work``, ``education``, …), or
- Legacy ``ResumeInfo`` flat fields from ``cv_agent`` extraction (both the
  older flat-string education/leadership shape and the structured
  EducationEntry / LeadershipEntry shape).

The output of ``build_harvard_sections`` feeds ``cv.tex.j2``:
- header dict: name / label / summary / contact_parts
- sections list: {"type", "title", "style", "entries"} where style is either
  "entries" (each entry is a list of rows ``{left, right, bold}`` rendered as
  ``left \\hfill right`` lines, plus an optional bullet list) or "itemize"
  (one compact bullet per entry with an optional bold label, used for skills
  and projects).

Every printed string is joined from segments that know the CV-data field they
came from. ``build_harvard_view`` returns the same sections with the segments
kept, for the CV editor, which shows the CV as it prints and writes each edit
back to its field. tests/golden/cv_tex.json pins the LaTeX output.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.services.cv_extraction_checks import repeats_grade, same_text

SECTION_LABELS: dict[str, str] = {
    "education": "Education",
    "work": "Experience",
    "volunteer": "Leadership & Activities",
    "projects": "Projects",
    "skills": "Skills",
    "awards": "Awards",
    "certificates": "Certifications",
    "publications": "Publications",
    "languages": "Languages",
    "interests": "Interests",
}

# Section titles for a CV whose content is in German.
SECTION_LABELS_DE: dict[str, str] = {
    "education": "Ausbildung",
    "work": "Berufserfahrung",
    "volunteer": "Engagement",
    "projects": "Projekte",
    "skills": "Kenntnisse",
    "awards": "Auszeichnungen",
    "certificates": "Zertifikate",
    "publications": "Veröffentlichungen",
    "languages": "Sprachen",
    "interests": "Interessen",
}

# Words the adapter itself writes into the CV, per content language.
_PHRASES = {
    "english": {"grade": "Grade", "technologies": "Technologies", "present": "present", "skills": "Skills",
                "languages": "Languages"},
    "german": {"grade": "Note", "technologies": "Technologien", "present": "heute", "skills": "Kenntnisse",
               "languages": "Sprachen"},
}

_MONTH_NUMBERS = {
    name: n
    for n, names in enumerate((
        ("jan", "january", "januar", "jän", "jänner"), ("feb", "february", "februar"),
        ("mar", "march", "mär", "märz", "maerz"), ("apr", "april"), ("may", "mai"),
        ("jun", "june", "juni"), ("jul", "july", "juli"), ("aug", "august"),
        ("sep", "sept", "september"), ("oct", "october", "okt", "oktober"),
        ("nov", "november"), ("dec", "december", "dez", "dezember"),
    ), start=1)
    for name in names
}
_MONTH_YEAR_RE = re.compile(r"\b([A-Za-zÄÖÜäöü]{3,9})\.?\s+((?:19|20)\d{2})\b")
_PRESENT_RE = re.compile(r"\b(present|current|now|today|ongoing|heute|aktuell|jetzt|dato)\b", re.IGNORECASE)
_RANGE_DASH_RE = re.compile(r"\s*(?:--|–|—|-|\bto\b|\bbis\b)\s*")


def format_dates_dach(text: str, language: str = "english") -> str:
    """'Jan 2022 -- Present' -> '01/2022 – present' ('heute' in German).
    Text without a recognisable month and year is left as it is."""
    if not text:
        return text

    def month_year(match: "re.Match[str]") -> str:
        month = _MONTH_NUMBERS.get(match.group(1).casefold())
        return f"{month:02d}/{match.group(2)}" if month else match.group(0)

    converted = _MONTH_YEAR_RE.sub(month_year, text)
    converted = _PRESENT_RE.sub(_PHRASES.get(language, _PHRASES["english"])["present"], converted)
    if converted == text:
        return text
    return _RANGE_DASH_RE.sub(" – ", converted, count=1)


# Skill-graph families by the CV skill group they belong to, for grouping a
# flat skills list when the strategist didn't group it.
_SKILL_CATEGORIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Programming languages", "Programmiersprachen",
     ("static_typed_systems_languages", "managed_compiled_languages", "dynamic_languages")),
    ("Web & backend", "Web & Backend",
     ("frontend_frameworks", "frontend_meta_frameworks", "css_frameworks", "state_management_js",
      "backend_python", "backend_node", "backend_jvm", "backend_go", "backend_ruby", "backend_dotnet",
      "graphql_servers", "auth_protocols", "auth_services", "build_tools_js", "package_managers_js",
      "mobile_native", "mobile_cross_platform", "sql_databases", "lite_sql", "nosql_document",
      "nosql_keyvalue", "search_engines", "message_queues", "streaming")),
    ("ML & data", "ML & Daten",
     ("deep_learning_frameworks", "classical_ml", "llm_tooling", "vector_databases", "data_processing",
      "distributed_compute", "data_orchestration", "etl_tools", "bi_tools", "analytics_warehouses",
      "lakehouse_table_formats")),
    ("Tools & infrastructure", "Tools & Infrastruktur",
     ("cloud_providers", "iaas_compute", "serverless_compute", "containers", "container_orchestration",
      "k8s_tooling", "iac", "ci_cd", "observability_metrics", "observability_tracing", "logging_stack",
      "testing_js", "testing_e2e", "testing_python")),
)
_OTHER_SKILLS = {"english": "Other", "german": "Weitere"}


# Words CV skill-category labels are made of ("Backend", "Frontend & Tools",
# "Programming Languages"), as opposed to skill names.
_CATEGORY_WORD_RE = re.compile(
    r"^(?:(?:backend|frontend|tools?|programming|languages?|frameworks?|libraries|databases?|devops|cloud|"
    r"ml|ai|data|web|mobile|testing|other|platforms?|agents?|infrastructure|technologies)"
    r"(?:\s*(?:&|and|/|,)\s*|\s+|$))+$",
    re.IGNORECASE,
)


def normalize_skill_list(skills: Any) -> list[str]:
    """
    Skill names only, each once. CV extraction sometimes keeps the CV's own
    category labels inside the items ('ML / Agent Frameworks: LangChain') and
    splits a label at its comma ('Backend', 'Frontend & Tools: FastAPI');
    the labels would otherwise print as skills.
    """
    from app.services import skill_graph  # local: keeps the adapter's import light

    items = [s.strip() for s in skills or [] if isinstance(s, str) and s.strip()]
    names: list[str] = []
    for i, item in enumerate(items):
        if ":" in item:
            item = item.split(":", 1)[1].strip()
        elif (i + 1 < len(items) and ":" in items[i + 1]
              and _CATEGORY_WORD_RE.match(item) and not skill_graph.find_family(item)):
            continue  # the first half of a label split at its comma
        if item and item.casefold() not in {n.casefold() for n in names}:
            names.append(item)
    return names


def group_skills(skills: list[str], language: str = "english") -> list[dict[str, Any]]:
    """
    A flat skills list as labelled groups, by skill-graph family. Skills the
    graph doesn't know go into a last 'Other' group. One group only (say,
    every skill unknown) gets no label, so the section doesn't read
    'Skills / Skills: ...'.
    """
    from app.services import skill_graph  # local: the adapter is imported by the renderer only

    german = language == "german"
    family_to_group = {family: (de if german else en) for en, de, families in _SKILL_CATEGORIES for family in families}
    groups: dict[str, list[str]] = {}
    for skill in skills:
        if not isinstance(skill, str) or not skill.strip():
            continue
        label = family_to_group.get(skill_graph.find_family(skill) or "", _OTHER_SKILLS["german" if german else "english"])
        groups.setdefault(label, []).append(skill.strip())
    order = [de if german else en for en, de, _ in _SKILL_CATEGORIES] + list(_OTHER_SKILLS.values())
    ordered = sorted(groups.items(), key=lambda item: order.index(item[0]))
    if len(ordered) == 1:
        return [{"name": "", "keywords": ordered[0][1]}]
    return [{"name": label, "keywords": names} for label, names in ordered]


DEFAULT_SECTION_ORDER: list[str] = [
    "education",
    "skills",
    "projects",
    "volunteer",
    "work",
    "awards",
    "certificates",
    "publications",
    "languages",
    "interests",
]

# Sections rendered as a compact itemize list instead of line-based entries.
_ITEMIZE_SECTIONS = {"skills", "projects", "languages", "interests"}


def _date_range(start: str | None, end: str | None, fallback: str | None = None) -> str:
    if fallback and str(fallback).strip():
        return str(fallback).strip()
    start = (start or "").strip()
    end = (end or "").strip()
    if start and end:
        return f"{start} -- {end}"
    return start or end


def _location_text(location: Any) -> str:
    if isinstance(location, str):
        return location.strip()
    if not isinstance(location, dict):
        return ""
    parts = [
        location.get("city"),
        location.get("region"),
        location.get("countryCode"),
        location.get("address"),
    ]
    return ", ".join(p for p in parts if isinstance(p, str) and p.strip())


def _is_json_resume(doc: dict[str, Any]) -> bool:
    return isinstance(doc.get("basics"), dict)


def _first(item: dict[str, Any], *keys: str) -> tuple[Any, str]:
    """The first of ``keys`` with a truthy value (as ``item.get(a) or item.get(b)``
    would pick it) and the key it came from; the first key when none has one,
    so an edit to an empty field lands on the canonical key."""
    for key in keys:
        if item.get(key):
            return item[key], key
    return None, keys[0]


def _sourced(items: Any, section: str) -> list[Any]:
    """Copies of a list's dict entries marked with where they came from
    (``section.i``); the caller's dicts are never modified."""
    return [{**item, "_src": f"{section}.{i}"} if isinstance(item, dict) else item
            for i, item in enumerate(items or [])]


def _legacy_to_json_resume(cv_data: dict[str, Any], language: str = "english") -> dict[str, Any]:
    """
    Map ResumeInfo-shaped dict into JSON Resume document fields.

    Each converted entry carries ``_src`` (its path in ``cv_data``, e.g.
    ``experience.0``) and ``_keys`` (which ``cv_data`` key each converted field
    was read from), so every printed piece of text can name the field it came
    from (see ``_path``) and the CV editor can write an edit back to it.
    """
    location = cv_data.get("location")
    basics_location: dict[str, Any] | str = {}
    if isinstance(location, str) and location.strip():
        basics_location = {"city": location.strip()}
    elif isinstance(location, dict):
        basics_location = location

    profiles: list[dict[str, Any]] = []
    for network, key in (
        ("LinkedIn", "linkedin_url"),
        ("GitHub", "github_url"),
        ("Portfolio", "portfolio_url"),
    ):
        url = cv_data.get(key)
        if isinstance(url, str) and url.strip():
            profiles.append({"network": network, "url": url.strip(), "_key": key})

    work: list[dict[str, Any]] = []
    for i, item in enumerate(cv_data.get("experience") or []):
        if isinstance(item, str):
            work.append({"name": "", "position": item, "highlights": [],
                         "_src": f"experience.{i}", "_keys": {"position": ""}})
            continue
        if not isinstance(item, dict):
            continue
        highlights, highlights_key = _first(item, "responsibilities", "highlights")
        duration, duration_key = _first(item, "duration", "dates")
        work.append(
            {
                "name": item.get("company") or "",
                "position": item.get("position") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "summary": item.get("summary") or "",
                "highlights": highlights or [],
                "_duration": duration or "",
                "_location": item.get("location") or "",
                "_src": f"experience.{i}",
                "_keys": {"name": "company", "highlights": highlights_key, "_duration": duration_key,
                          "_location": "location"},
            }
        )

    education: list[dict[str, Any]] = []
    for i, item in enumerate(cv_data.get("education") or []):
        if isinstance(item, str):
            education.append({"institution": "", "area": "", "studyType": "", "_text": item.strip(),
                              "_src": f"education.{i}", "_keys": {"_text": ""}})
            continue
        if not isinstance(item, dict):
            continue
        institution, institution_key = _first(item, "institution", "school")
        area, area_key = _first(item, "area", "degree")
        score, score_key = _first(item, "score", "grade", "gpa")
        education.append(
            {
                "institution": institution or "",
                "area": area or "",
                "studyType": item.get("studyType") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "score": score or "",
                "courses": item.get("courses") or [],
                "_text": item.get("text") or "",
                "_location": item.get("location") or "",
                "_dates": item.get("dates") or "",
                "_details": item.get("details") or [],
                "_src": f"education.{i}",
                "_keys": {"institution": institution_key, "area": area_key, "score": score_key, "_text": "text",
                          "_location": "location", "_dates": "dates", "_details": "details"},
            }
        )

    volunteer: list[dict[str, Any]] = []
    volunteer_raw, volunteer_section = _first(cv_data, "volunteer", "leadership_activities")
    for i, item in enumerate(volunteer_raw or []):
        if isinstance(item, str):
            volunteer.append({"organization": "", "position": item, "highlights": [],
                              "_src": f"{volunteer_section}.{i}", "_keys": {"position": ""}})
            continue
        if not isinstance(item, dict):
            continue
        organization, organization_key = _first(item, "organization", "company")
        position, position_key = _first(item, "position", "role")
        summary, summary_key = _first(item, "description", "summary")
        duration, duration_key = _first(item, "dates", "duration")
        volunteer.append(
            {
                "organization": organization or "",
                "position": position or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "summary": summary or "",
                "highlights": item.get("highlights") or [],
                "_duration": duration or "",
                "_location": item.get("location") or "",
                "_src": f"{volunteer_section}.{i}",
                "_keys": {"organization": organization_key, "position": position_key, "summary": summary_key,
                          "_duration": duration_key, "_location": "location"},
            }
        )

    projects: list[dict[str, Any]] = []
    for i, item in enumerate(cv_data.get("projects") or []):
        if isinstance(item, str):
            projects.append({"name": item, "description": "", "highlights": [],
                             "_src": f"projects.{i}", "_keys": {"name": ""}})
            continue
        if not isinstance(item, dict):
            continue
        highlights, highlights_key = _first(item, "highlights", "outcomes")
        technologies = item.get("technologies") or []
        tech_text = ""
        if isinstance(technologies, list) and technologies:
            tech_text = ", ".join(t for t in technologies if isinstance(t, str) and t.strip())
        elif isinstance(technologies, str):
            tech_text = technologies.strip()
        projects.append(
            {
                "name": item.get("name") or "",
                "description": item.get("description") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "url": item.get("url") or "",
                "highlights": list(highlights or []),
                "_technologies": tech_text,
                "_src": f"projects.{i}",
                "_keys": {"highlights": highlights_key, "_technologies": "technologies"},
                "_list_fields": ("_technologies",) if isinstance(technologies, list) else (),
            }
        )

    skills_raw = cv_data.get("skills") or []
    skills: list[dict[str, Any]] = []
    if skills_raw:
        if all(isinstance(s, str) for s in skills_raw):
            # Grouped here, so the groups have no field of their own to edit.
            skills = group_skills(skills_raw, language)
        elif all(isinstance(s, dict) for s in skills_raw):
            skills = _sourced(skills_raw, "skills")
        else:
            # Mixed content: keep strings under a generic group, dicts as-is.
            flat = [s for s in skills_raw if isinstance(s, str) and s.strip()]
            skills = [s for s in _sourced(skills_raw, "skills") if isinstance(s, dict)]
            if flat:
                skills.append({"name": "Skills", "keywords": flat})

    url, url_key = _first(cv_data, "portfolio_url", "url")
    doc: dict[str, Any] = {
        "basics": {
            "name": cv_data.get("name") or "Applicant",
            "label": cv_data.get("label") or "",
            "email": cv_data.get("email") or "",
            "phone": cv_data.get("phone") or "",
            "url": url or "",
            "summary": cv_data.get("summary") or "",
            "location": basics_location,
            "profiles": profiles,
            "_keys": {"url": url_key},
        },
        "work": work,
        "education": education,
        "volunteer": volunteer,
        "projects": projects,
        "skills": skills,
        "awards": _sourced(cv_data.get("awards"), "awards"),
        "certificates": _sourced(cv_data.get("certificates"), "certificates"),
        "publications": _sourced(cv_data.get("publications"), "publications"),
        "languages": _sourced(cv_data.get("languages"), "languages"),
        "interests": _sourced(cv_data.get("interests"), "interests"),
    }
    return doc


def to_json_resume(cv_data: dict[str, Any], language: str = "english") -> dict[str, Any]:
    """Normalize any supported CV dict into JSON Resume document shape."""
    if _is_json_resume(cv_data):
        return cv_data
    return _legacy_to_json_resume(cv_data, language)


def _bullets(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


# --------------------------------------------------------------------------- #
# Segments: printed text that knows where it came from
#
# Every row, bullet and itemize entry is built as a list of segments
# {"text", "src", "value"?, "kind"?}. ``text`` is what prints; ``src`` the
# paths in the CV data it came from (empty for the adapter's own words, e.g.
# "Technologies: " or a separator); ``value`` the field's raw value where the
# printed text is formatted (DACH dates); ``kind`` "list" for a comma-joined
# list field. The strings the template prints are joined from the segments as
# the last step (_finalize), so the editor's view and the PDF can't disagree.
# --------------------------------------------------------------------------- #

def _seg(text: str, src: list[str] | None = None, kind: str | None = None) -> dict[str, Any]:
    seg: dict[str, Any] = {"text": text, "src": list(src or [])}
    if kind:
        seg["kind"] = kind
    return seg


def _static(text: str) -> dict[str, Any]:
    """The adapter's own words or punctuation: printed, not editable."""
    return _seg(text)


def _text(segs: list[dict[str, Any]]) -> str:
    return "".join(s["text"] for s in segs)


def _retext(seg: dict[str, Any], text: str) -> None:
    """Change what a segment prints (formatting), keeping its raw value for editing."""
    if text != seg["text"]:
        if seg["src"] and "value" not in seg:
            seg["value"] = seg["text"]
        seg["text"] = text


def _path(item: dict[str, Any], field: str, *index: int) -> list[str]:
    """The CV-data path a converted entry's field was read from (``experience.0.company``),
    or none for data without a source (JSON Resume input, derived groups)."""
    base = item.get("_src")
    if base is None:
        return []
    key = (item.get("_keys") or {}).get(field, field)
    return [".".join([base, *([key] if key else []), *(str(i) for i in index)])]


def _field_seg(item: dict[str, Any], field: str, text: str) -> list[dict[str, Any]]:
    """One editable segment for a field, or nothing when it prints nothing."""
    if not text:
        return []
    kind = "list" if field in (item.get("_list_fields") or ()) else None
    return [_seg(text, _path(item, field), kind)]


def _bullet_segs(item: dict[str, Any], field: str) -> list[list[dict[str, Any]]]:
    """A list field's non-empty items, each as a segment pointing at its own index."""
    value = item.get(field)
    if not isinstance(value, list):
        return []
    return [[_seg(v.strip(), _path(item, field, j))] for j, v in enumerate(value)
            if isinstance(v, str) and v.strip()]


def _joined(parts: list[list[dict[str, Any]]], separator: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in parts:
        if out:
            out.append(_static(separator))
        out.extend(part)
    return out


# --------------------------------------------------------------------------- #
# Entry shapes for the template
# --------------------------------------------------------------------------- #

def _row(left: list[dict[str, Any]], right: list[dict[str, Any]] | None = None, bold: bool = False) -> dict[str, Any]:
    """One rendered line: ``left \\hfill right`` (left bolded when bold)."""
    return {"left": left, "right": right or [], "bold": bold}


def _entry(rows: list[dict[str, Any]] | None = None, bullets: list[list[dict[str, Any]]] | None = None,
           bullets_src: list[str] | None = None) -> dict[str, Any]:
    """Rows plus a bullet list; ``bullets_src`` is the list's own path, for
    adding, removing and moving bullets in the editor."""
    return {"rows": rows or [], "bullets": bullets or [], "bullets_src": list(bullets_src or [])}


def _item(label: list[dict[str, Any]], text: list[dict[str, Any]]) -> dict[str, Any]:
    """One compact itemize entry: optional bold label + text."""
    return {"label": label, "text": text}


def _dates_segs(item: dict[str, Any], start_field: str = "startDate", end_field: str = "endDate",
                fallback_field: str = "_duration") -> list[dict[str, Any]]:
    """The entry's date range as printed (_date_range), editable as the
    fallback field (``duration`` / ``dates``), which the range prefers."""
    fallback = item.get(fallback_field)
    text = _date_range(item.get(start_field), item.get(end_field), fallback)
    return [_seg(text, _path(item, fallback_field))] if text else []


def _location_segs(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item.get("_location"):
        return _field_seg(item, "_location", item["_location"])
    text = _location_text(item.get("location"))
    return [_seg(text, _path(item, "location"))] if text else []


def _rows_for_work(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dates = _dates_segs(item)
        location = _location_segs(item)
        company = _field_seg(item, "name", item.get("name") or "")
        position = _field_seg(item, "position", item.get("position") or "")
        if same_text(_text(company), _text(position)):
            # The title read twice (no employer named), not a company: print it once.
            company = []

        rows: list[dict[str, Any]] = []
        if company:
            rows.append(_row(company, location, bold=True))
            if position or dates:
                rows.append(_row(position, dates, bold=True))
        elif position:
            rows.append(_row(position, dates or location, bold=True))

        summary = item.get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(_field_seg(item, "summary", summary.strip())))

        entries.append(_entry(rows=rows, bullets=_bullet_segs(item, "highlights"),
                              bullets_src=_path(item, "highlights")))
    return entries


def _rows_for_education(items: list[Any]) -> list[dict[str, Any]]:
    """Group consecutive entries of the same institution under one bold
    header row, mirroring the Harvard layout:

        **Institution** \\hfill Location
        Degree \\hfill Dates
        Grade: ...
        Degree 2 \\hfill Dates 2
    """
    entries: list[dict[str, Any]] = []
    current_institution: str | None = None
    current_entry: dict[str, Any] | None = None

    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("_text"):
            entries.append(_entry(rows=[_row(_field_seg(item, "_text", str(item["_text"])))]))
            current_institution, current_entry = None, None
            continue

        institution = (item.get("institution") or "").strip()
        location = _location_segs(item)
        # The written dates as they are; else a start/end range, editable as ``dates``.
        dates = (_field_seg(item, "_dates", item["_dates"]) if item.get("_dates")
                 else _dates_segs(item, fallback_field="_dates"))
        study = _joined([_field_seg(item, field, p) for field in ("studyType", "area")
                         if isinstance(p := item.get(field), str) and p.strip()], " ")

        detail_rows: list[dict[str, Any]] = []
        score = item.get("score")
        if isinstance(score, str) and score.strip():
            grade = score.strip()
            grade_segs = _field_seg(item, "score", grade)
            if not re.search(r"\b(grade|gpa|note|notendurchschnitt|durchschnitt)\b", grade, re.IGNORECASE):
                grade_segs = [_static("Grade: "), *grade_segs]
            detail_rows.append(_row(grade_segs))
        for field in ("_details", "courses"):
            for segs in _bullet_segs(item, field):
                if field == "courses" or not repeats_grade(_text(segs), str(score or "")):
                    detail_rows.append(_row(segs))

        if institution and current_entry is not None and institution == current_institution:
            # Same school: append the degree lines to the open entry; the shared
            # header row now edits both entries' institution.
            current_entry["rows"][0]["left"][0]["src"].extend(_path(item, "institution"))
            current_entry["rows"].append(_row(study, dates))
            current_entry["rows"].extend(detail_rows)
            continue

        rows: list[dict[str, Any]] = []
        if institution:
            rows.append(_row(_field_seg(item, "institution", institution), location, bold=True))
        if study or dates:
            rows.append(_row(study, dates))
        rows.extend(detail_rows)

        current_entry = _entry(rows=rows)
        current_institution = institution or None
        entries.append(current_entry)

    return entries


def _rows_for_volunteer(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dates = _dates_segs(item)
        organization = _field_seg(item, "organization", item.get("organization") or "")
        position = _field_seg(item, "position", item.get("position") or "")
        if same_text(_text(organization), _text(position)):
            organization = []
        location = _location_segs(item)
        summary = item.get("summary") or ""
        bullets = _bullet_segs(item, "highlights")

        # Legacy flat-string activities: a bare position with no structure.
        if position and not organization and not dates and not bullets and not summary:
            entries.append(_entry(rows=[_row(position)]))
            continue

        rows: list[dict[str, Any]] = []
        if organization:
            rows.append(_row(organization, location, bold=True))
            if position or dates:
                rows.append(_row(position, dates, bold=True))
        elif position:
            rows.append(_row(position, dates or location, bold=True))

        if isinstance(summary, str) and summary.strip():
            rows.append(_row(_field_seg(item, "summary", summary.strip())))

        entries.append(_entry(rows=rows, bullets=bullets, bullets_src=_path(item, "highlights")))
    return entries


def _project_text(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold a project's description, technologies, and highlights into one
    compact paragraph for the itemize entry."""
    parts: list[list[dict[str, Any]]] = []
    description = item.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(_field_seg(item, "description", description.strip()))
    tech = item.get("_technologies") or ""
    if isinstance(tech, str) and tech.strip():
        parts.append([_static("Technologies: "), *_field_seg(item, "_technologies", tech.strip()), _static(".")])
    for segs in _bullet_segs(item, "highlights"):
        highlight = _text(segs)
        # Skip a redundant "Technologies: ..." highlight from older data.
        if highlight.lower().startswith("technologies:") and tech:
            continue
        parts.append(segs if highlight.endswith((".", "!", "?")) else [*segs, _static(".")])
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        parts.append(_field_seg(item, "url", url.strip()))
    return _joined(parts, " ")


def _rows_for_projects(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(_item(label=_field_seg(item, "name", item.get("name") or ""), text=_project_text(item)))
    return rows


def _rows_for_skills(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            rows.append(_item(label=[], text=[_static(item.strip())]))
            continue
        if not isinstance(item, dict):
            continue
        keywords = ", ".join(_bullets(item.get("keywords")))
        parts = [part for part in (
            [_seg(keywords, _path(item, "keywords"), "list")] if keywords else [],
            _field_seg(item, "level", item.get("level") or ""),
        ) if part]
        # A group without a name (the only group) prints without a label.
        rows.append(_item(label=_field_seg(item, "name", item.get("name") or ""), text=_joined(parts, " | ")))
    return rows


def _rows_for_awards(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(_field_seg(item, "title", item.get("title") or ""),
                     _field_seg(item, "date", item.get("date") or ""), bold=True)]
        if item.get("awarder"):
            rows.append(_row(_field_seg(item, "awarder", item["awarder"])))
        summary = item.get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(_field_seg(item, "summary", summary.strip())))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_certificates(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(_field_seg(item, "name", item.get("name") or ""),
                     _field_seg(item, "date", item.get("date") or ""), bold=True)]
        if item.get("issuer"):
            rows.append(_row(_field_seg(item, "issuer", item["issuer"])))
        url = item.get("url") or ""
        if isinstance(url, str) and url.strip():
            rows.append(_row(_field_seg(item, "url", url.strip())))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_publications(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(_field_seg(item, "name", item.get("name") or ""),
                     _field_seg(item, "releaseDate", item.get("releaseDate") or ""), bold=True)]
        if item.get("publisher"):
            rows.append(_row(_field_seg(item, "publisher", item["publisher"])))
        summary, summary_key = _first(item, "summary", "url")
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(_field_seg(item, summary_key, summary.strip())))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_languages(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(_item(label=_field_seg(item, "language", item.get("language") or ""),
                          text=_field_seg(item, "fluency", item.get("fluency") or "")))
    return rows


def _rows_for_interests(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keywords = ", ".join(_bullets(item.get("keywords")))
        rows.append(_item(label=_field_seg(item, "name", item.get("name") or ""),
                          text=[_seg(keywords, _path(item, "keywords"), "list")] if keywords else []))
    return rows


_ROW_BUILDERS = {
    "work": _rows_for_work,
    "education": _rows_for_education,
    "volunteer": _rows_for_volunteer,
    "projects": _rows_for_projects,
    "skills": _rows_for_skills,
    "awards": _rows_for_awards,
    "certificates": _rows_for_certificates,
    "publications": _rows_for_publications,
    "languages": _rows_for_languages,
    "interests": _rows_for_interests,
}


def _printable(text: Any) -> str:
    """Text without the invisible characters PDF extraction picks up: icon-font
    glyphs (private use), control and format characters. 'Phone' and 'LinkedIn'
    icons in a CV template otherwise come through as values that print as
    nothing, leaving empty slots in the contact line."""
    if not isinstance(text, str):
        return ""
    cleaned = "".join(ch for ch in text if unicodedata.category(ch) not in ("Co", "Cc", "Cf", "Cs", "Cn"))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if any(ch.isalnum() for ch in cleaned) else ""


def _basics_contact_parts(basics: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Contact fragments rendered as ``\\textbullet``-separated header parts:
    location first, then email/phone, then plain profile URLs. Empty or
    invisible values are skipped and repeats dropped, so no slot is blank and
    nothing appears twice."""
    sourced = "_keys" in basics  # converted from CV data, so each part has a field
    candidates = [
        (_location_text(basics.get("location")), "location"),
        (basics.get("email"), "email"),
        (basics.get("phone"), "phone"),
        (basics.get("url"), (basics.get("_keys") or {}).get("url", "url")),
    ]
    candidates += [(p.get("url"), p.get("_key", "")) for p in basics.get("profiles") or [] if isinstance(p, dict)]
    parts: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for value, field in candidates:
        part = _printable(value)
        key = part.casefold().rstrip("/")
        if part and key not in seen:
            seen.add(key)
            parts.append([_seg(part, [field] if sourced and field else [])])
    return parts


def _resolve_section_order(section_order: list[str] | None) -> list[str]:
    if not section_order:
        return DEFAULT_SECTION_ORDER
    mapped: list[str] = []
    alias = {
        "experience": "work",
        "leadership": "volunteer",
        "leadership_activities": "volunteer",
        "skills_projects": "skills",
    }
    for name in section_order:
        key = alias.get(name, name)
        if key in SECTION_LABELS and key not in mapped:
            mapped.append(key)
    for key in DEFAULT_SECTION_ORDER:
        if key not in mapped:
            mapped.append(key)
    return mapped


def _localize_entries(entries: list[dict[str, Any]], language: str, convention: str) -> list[dict[str, Any]]:
    """German wording for the adapter's own words, and DACH date style. Works
    on the segments, so a formatted date keeps its raw value for editing."""
    if language != "german" and convention != "dach":
        return entries
    phrases = _PHRASES["german"] if language == "german" else _PHRASES["english"]
    out = []
    for entry in entries:
        entry = dict(entry)
        if "rows" in entry:
            rows = []
            for row in entry["rows"]:
                row = {**row, "left": [dict(s) for s in row["left"]], "right": [dict(s) for s in row["right"]]}
                if language == "german" and row["left"] and row["left"][0]["text"].startswith("Grade: "):
                    first = row["left"][0]
                    _retext(first, f"{phrases['grade']}: " + first["text"][len("Grade: "):])
                if convention == "dach":
                    for seg in row["right"]:
                        _retext(seg, format_dates_dach(seg["text"], language))
                rows.append(row)
            entry["rows"] = rows
        if language == "german" and "text" in entry:
            entry["text"] = [dict(s) for s in entry["text"]]
            for seg in entry["text"]:
                _retext(seg, seg["text"].replace("Technologies: ", f"{phrases['technologies']}: "))
        out.append(entry)
    return out


def _merge_languages_into_skills(sections: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Fold a short Languages section into Skills as one labelled line, so it
    can't end up alone on the last page."""
    by_type = {s["type"]: s for s in sections}
    languages, skills = by_type.get("languages"), by_type.get("skills")
    if not languages or not skills:
        return sections
    spoken = _joined([
        [*e["label"], _static(" ("), *e["text"], _static(")")] if _text(e["text"]) else e["label"]
        for e in languages["entries"] if _text(e["label"])
    ], ", ")
    merged = {**skills, "entries": skills["entries"] + [_item([_static(_PHRASES[language]["languages"])], spoken)]}
    return [merged if s is skills else s for s in sections if s is not languages]


def _finalize(entries: list[dict[str, Any]], keep_segments: bool) -> None:
    """Join each entry's segments into the strings the template prints; with
    ``keep_segments`` the segments stay next to them (``left_segs``,
    ``bullet_segs``, ``text_segs`` ...) for the editor's view."""
    for entry in entries:
        if "rows" in entry:
            for row in entry["rows"]:
                left, right = row["left"], row["right"]
                row["left"], row["right"] = _text(left), _text(right)
                if keep_segments:
                    row["left_segs"], row["right_segs"] = left, right
            bullets = entry["bullets"]
            entry["bullets"] = [_text(segs) for segs in bullets]
            if keep_segments:
                entry["bullet_segs"] = bullets
            else:
                del entry["bullets_src"]
        else:
            label, text = entry["label"], entry["text"]
            entry["label"], entry["text"] = _text(label), _text(text)
            if keep_segments:
                entry["label_segs"], entry["text_segs"] = label, text


def build_harvard_sections(
    cv_data: dict[str, Any],
    section_order: list[str] | None = None,
    *,
    language: str = "english",
    convention: str = "international",
    merge_languages: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Build template context: basics header fields + ordered sections.

    Each section is ``{"type", "title", "style", "entries"}`` where style is
    "entries" (row-based blocks) or "itemize" (compact labelled bullets).

    ``language`` is the language the CV content is written in (section titles
    follow it); ``convention`` "dach" writes dates as MM/YYYY. Spoken
    languages become a line in Skills unless ``merge_languages`` is False
    (or there is no Skills section).

    The strings are joined from segments that know their source field; see
    ``build_harvard_view`` for the same sections with the segments kept.
    """
    return _build(cv_data, section_order, language=language, convention=convention,
                  merge_languages=merge_languages, keep_segments=False)


def build_harvard_view(
    cv_data: dict[str, Any],
    section_order: list[str] | None = None,
    *,
    language: str = "english",
    convention: str = "international",
    merge_languages: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    The CV editor's view: ``build_harvard_sections`` with every printed string's
    segments kept (``name_segs``, ``contact_segs``, rows' ``left_segs`` /
    ``right_segs``, ``bullet_segs`` with the list's ``bullets_src``, itemize
    ``label_segs`` / ``text_segs``). The editor shows these as the PDF prints
    them and writes each edit back to the segment's ``src``.
    """
    return _build(cv_data, section_order, language=language, convention=convention,
                  merge_languages=merge_languages, keep_segments=True)


def _build(
    cv_data: dict[str, Any],
    section_order: list[str] | None,
    *,
    language: str,
    convention: str,
    merge_languages: bool,
    keep_segments: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    language = language if language in _PHRASES else "english"
    labels = SECTION_LABELS_DE if language == "german" else SECTION_LABELS
    doc = to_json_resume(cv_data, language)
    sourced = not _is_json_resume(cv_data)
    basics = doc.get("basics") or {}
    label, summary = basics.get("label") or "", basics.get("summary") or ""
    if summary.strip():
        # With a summary, a headline only restates it (often as a paraphrase
        # of its first sentence), so the intro read twice. It is printed only
        # for a CV without a summary.
        label = ""
    name = basics.get("name") or cv_data.get("name") or "Applicant"
    contact_segs = _basics_contact_parts(basics)
    header = {
        "name": name,
        "label": label,
        "summary": summary,
        "contact_parts": [_text(segs) for segs in contact_segs],
    }
    if keep_segments:
        header.update({
            "name_segs": [_seg(name, ["name"] if sourced else [])],
            "label_segs": [_seg(label, ["label"] if sourced else [])] if label else [],
            "summary_segs": [_seg(summary, ["summary"] if sourced else [])] if summary else [],
            "contact_segs": contact_segs,
        })

    order = _resolve_section_order(section_order)
    sections: list[dict[str, Any]] = []
    for section_type in order:
        items = doc.get(section_type) or []
        if not items:
            continue
        builder = _ROW_BUILDERS.get(section_type)
        if not builder:
            continue
        entries = builder(items)
        if not entries:
            continue
        sections.append(
            {
                "type": section_type,
                "title": labels[section_type],
                "style": "itemize" if section_type in _ITEMIZE_SECTIONS else "entries",
                "entries": _localize_entries(entries, language, convention),
            }
        )
    if merge_languages:
        sections = _merge_languages_into_skills(sections, language)
    for section in sections:
        _finalize(section["entries"], keep_segments)
    return header, sections
