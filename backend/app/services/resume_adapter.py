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


def _legacy_to_json_resume(cv_data: dict[str, Any], language: str = "english") -> dict[str, Any]:
    """Map ResumeInfo-shaped dict into JSON Resume document fields."""
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
            profiles.append({"network": network, "url": url.strip()})

    work: list[dict[str, Any]] = []
    for item in cv_data.get("experience") or []:
        if isinstance(item, str):
            work.append({"name": "", "position": item, "highlights": []})
            continue
        if not isinstance(item, dict):
            continue
        work.append(
            {
                "name": item.get("company") or "",
                "position": item.get("position") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "summary": item.get("summary") or "",
                "highlights": item.get("responsibilities")
                or item.get("highlights")
                or [],
                "_duration": item.get("duration") or item.get("dates") or "",
                "_location": item.get("location") or "",
            }
        )

    education: list[dict[str, Any]] = []
    for item in cv_data.get("education") or []:
        if isinstance(item, str):
            education.append({"institution": "", "area": "", "studyType": "", "_text": item.strip()})
            continue
        if not isinstance(item, dict):
            continue
        education.append(
            {
                "institution": item.get("institution") or item.get("school") or "",
                "area": item.get("area") or item.get("degree") or "",
                "studyType": item.get("studyType") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "score": item.get("score") or item.get("grade") or item.get("gpa") or "",
                "courses": item.get("courses") or [],
                "_text": item.get("text") or "",
                "_location": item.get("location") or "",
                "_dates": item.get("dates") or "",
                "_details": item.get("details") or [],
            }
        )

    volunteer: list[dict[str, Any]] = []
    for item in cv_data.get("volunteer") or cv_data.get("leadership_activities") or []:
        if isinstance(item, str):
            volunteer.append({"organization": "", "position": item, "highlights": []})
            continue
        if not isinstance(item, dict):
            continue
        volunteer.append(
            {
                "organization": item.get("organization") or item.get("company") or "",
                "position": item.get("position") or item.get("role") or "",
                "startDate": item.get("startDate") or "",
                "endDate": item.get("endDate") or "",
                "summary": item.get("description") or item.get("summary") or "",
                "highlights": item.get("highlights") or [],
                "_duration": item.get("dates") or item.get("duration") or "",
                "_location": item.get("location") or "",
            }
        )

    projects: list[dict[str, Any]] = []
    for item in cv_data.get("projects") or []:
        if isinstance(item, str):
            projects.append({"name": item, "description": "", "highlights": []})
            continue
        if not isinstance(item, dict):
            continue
        highlights = list(item.get("highlights") or item.get("outcomes") or [])
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
                "highlights": highlights,
                "_technologies": tech_text,
            }
        )

    skills_raw = cv_data.get("skills") or []
    skills: list[dict[str, Any]] = []
    if skills_raw:
        if all(isinstance(s, str) for s in skills_raw):
            skills = group_skills(skills_raw, language)
        elif all(isinstance(s, dict) for s in skills_raw):
            skills = skills_raw
        else:
            # Mixed content: keep strings under a generic group, dicts as-is.
            flat = [s for s in skills_raw if isinstance(s, str) and s.strip()]
            skills = [s for s in skills_raw if isinstance(s, dict)]
            if flat:
                skills.append({"name": "Skills", "keywords": flat})

    doc: dict[str, Any] = {
        "basics": {
            "name": cv_data.get("name") or "Applicant",
            "label": cv_data.get("label") or "",
            "email": cv_data.get("email") or "",
            "phone": cv_data.get("phone") or "",
            "url": cv_data.get("portfolio_url") or cv_data.get("url") or "",
            "summary": cv_data.get("summary") or "",
            "location": basics_location,
            "profiles": profiles,
        },
        "work": work,
        "education": education,
        "volunteer": volunteer,
        "projects": projects,
        "skills": skills,
        "awards": cv_data.get("awards") or [],
        "certificates": cv_data.get("certificates") or [],
        "publications": cv_data.get("publications") or [],
        "languages": cv_data.get("languages") or [],
        "interests": cv_data.get("interests") or [],
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
# Entry shapes for the template
# --------------------------------------------------------------------------- #

def _row(left: str, right: str = "", bold: bool = False) -> dict[str, Any]:
    """One rendered line: ``left \\hfill right`` (left bolded when bold)."""
    return {"left": left, "right": right, "bold": bold}


def _entry(rows: list[dict[str, Any]] | None = None, bullets: list[str] | None = None) -> dict[str, Any]:
    return {"rows": rows or [], "bullets": bullets or []}


def _item(label: str, text: str) -> dict[str, Any]:
    """One compact itemize entry: optional bold label + text."""
    return {"label": label, "text": text}


def _rows_for_work(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dates = _date_range(
            item.get("startDate"),
            item.get("endDate"),
            item.get("_duration"),
        )
        location = item.get("_location") or _location_text(item.get("location"))
        company = item.get("name") or ""
        position = item.get("position") or ""
        if same_text(company, position):
            # The title read twice (no employer named), not a company: print it once.
            company = ""

        rows: list[dict[str, Any]] = []
        if company:
            rows.append(_row(company, location, bold=True))
            if position or dates:
                rows.append(_row(position, dates, bold=True))
        elif position:
            rows.append(_row(position, dates or location, bold=True))

        summary = item.get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(summary.strip()))

        entries.append(_entry(rows=rows, bullets=_bullets(item.get("highlights"))))
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
            entries.append(_entry(rows=[_row(str(item["_text"]))]))
            current_institution, current_entry = None, None
            continue

        institution = (item.get("institution") or "").strip()
        location = item.get("_location") or _location_text(item.get("location"))
        dates = item.get("_dates") or _date_range(item.get("startDate"), item.get("endDate"))
        study = " ".join(
            p for p in (item.get("studyType"), item.get("area")) if isinstance(p, str) and p.strip()
        )

        detail_rows: list[dict[str, Any]] = []
        score = item.get("score")
        if isinstance(score, str) and score.strip():
            grade = score.strip()
            if not re.search(r"\b(grade|gpa|note|notendurchschnitt|durchschnitt)\b", grade, re.IGNORECASE):
                grade = f"Grade: {grade}"
            detail_rows.append(_row(grade))
        for detail in _bullets(item.get("_details")):
            if not repeats_grade(detail, str(score or "")):
                detail_rows.append(_row(detail))
        for course in _bullets(item.get("courses")):
            detail_rows.append(_row(course))

        if institution and current_entry is not None and institution == current_institution:
            # Same school: append the degree lines to the open entry.
            current_entry["rows"].append(_row(study, dates))
            current_entry["rows"].extend(detail_rows)
            continue

        rows: list[dict[str, Any]] = []
        if institution:
            rows.append(_row(institution, location, bold=True))
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
        dates = _date_range(
            item.get("startDate"),
            item.get("endDate"),
            item.get("_duration"),
        )
        organization = item.get("organization") or ""
        position = item.get("position") or ""
        if same_text(organization, position):
            organization = ""
        location = item.get("_location") or _location_text(item.get("location"))
        summary = item.get("summary") or ""
        bullets = _bullets(item.get("highlights"))

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
            rows.append(_row(summary.strip()))

        entries.append(_entry(rows=rows, bullets=bullets))
    return entries


def _project_text(item: dict[str, Any]) -> str:
    """Fold a project's description, technologies, and highlights into one
    compact paragraph for the itemize entry."""
    parts: list[str] = []
    description = item.get("description")
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    tech = item.get("_technologies") or ""
    if isinstance(tech, str) and tech.strip():
        parts.append(f"Technologies: {tech.strip()}.")
    for highlight in _bullets(item.get("highlights")):
        # Skip a redundant "Technologies: ..." highlight from older data.
        if highlight.lower().startswith("technologies:") and tech:
            continue
        parts.append(highlight if highlight.endswith((".", "!", "?")) else highlight + ".")
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        parts.append(url.strip())
    return " ".join(parts)


def _rows_for_projects(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(_item(label=item.get("name") or "", text=_project_text(item)))
    return rows


def _rows_for_skills(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            rows.append(_item(label="", text=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        keywords = ", ".join(_bullets(item.get("keywords")))
        level = item.get("level") or ""
        text = " | ".join(p for p in (keywords, level) if p)
        # A group without a name (the only group) prints without a label.
        rows.append(_item(label=item.get("name") or "", text=text))
    return rows


def _rows_for_awards(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(item.get("title") or "", item.get("date") or "", bold=True)]
        if item.get("awarder"):
            rows.append(_row(item["awarder"]))
        summary = item.get("summary") or ""
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(summary.strip()))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_certificates(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(item.get("name") or "", item.get("date") or "", bold=True)]
        if item.get("issuer"):
            rows.append(_row(item["issuer"]))
        url = item.get("url") or ""
        if isinstance(url, str) and url.strip():
            rows.append(_row(url.strip()))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_publications(items: list[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows = [_row(item.get("name") or "", item.get("releaseDate") or "", bold=True)]
        if item.get("publisher"):
            rows.append(_row(item["publisher"]))
        summary = item.get("summary") or item.get("url") or ""
        if isinstance(summary, str) and summary.strip():
            rows.append(_row(summary.strip()))
        entries.append(_entry(rows=rows))
    return entries


def _rows_for_languages(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(_item(label=item.get("language") or "", text=item.get("fluency") or ""))
    return rows


def _rows_for_interests(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(_item(label=item.get("name") or "", text=", ".join(_bullets(item.get("keywords")))))
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


def _basics_contact_parts(basics: dict[str, Any]) -> list[str]:
    """Contact fragments rendered as ``\\textbullet``-separated header parts:
    location first, then email/phone, then plain profile URLs. Empty or
    invisible values are skipped and repeats dropped, so no slot is blank and
    nothing appears twice."""
    candidates = [_location_text(basics.get("location")), basics.get("email"), basics.get("phone"), basics.get("url")]
    candidates += [p.get("url") for p in basics.get("profiles") or [] if isinstance(p, dict)]
    parts: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        part = _printable(value)
        key = part.casefold().rstrip("/")
        if part and key not in seen:
            seen.add(key)
            parts.append(part)
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
    """German wording for the adapter's own words, and DACH date style."""
    if language != "german" and convention != "dach":
        return entries
    phrases = _PHRASES["german"] if language == "german" else _PHRASES["english"]
    out = []
    for entry in entries:
        entry = dict(entry)
        if "rows" in entry:
            rows = []
            for row in entry["rows"]:
                row = dict(row)
                if language == "german" and row["left"].startswith("Grade: "):
                    row["left"] = f"{phrases['grade']}: " + row["left"][len("Grade: "):]
                if convention == "dach":
                    row["right"] = format_dates_dach(row["right"], language)
                rows.append(row)
            entry["rows"] = rows
        if language == "german" and isinstance(entry.get("text"), str):
            entry["text"] = entry["text"].replace("Technologies: ", f"{phrases['technologies']}: ")
        out.append(entry)
    return out


def _merge_languages_into_skills(sections: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    """Fold a short Languages section into Skills as one labelled line, so it
    can't end up alone on the last page."""
    by_type = {s["type"]: s for s in sections}
    languages, skills = by_type.get("languages"), by_type.get("skills")
    if not languages or not skills:
        return sections
    spoken = ", ".join(
        f"{e['label']} ({e['text']})" if e.get("text") else e.get("label", "")
        for e in languages["entries"] if e.get("label")
    )
    merged = {**skills, "entries": skills["entries"] + [_item(_PHRASES[language]["languages"], spoken)]}
    return [merged if s is skills else s for s in sections if s is not languages]


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
    """
    language = language if language in _PHRASES else "english"
    labels = SECTION_LABELS_DE if language == "german" else SECTION_LABELS
    doc = to_json_resume(cv_data, language)
    basics = doc.get("basics") or {}
    label, summary = basics.get("label") or "", basics.get("summary") or ""
    if summary.strip():
        # With a summary, a headline only restates it (often as a paraphrase
        # of its first sentence), so the intro read twice. It is printed only
        # for a CV without a summary.
        label = ""
    header = {
        "name": basics.get("name") or cv_data.get("name") or "Applicant",
        "label": label,
        "summary": summary,
        "contact_parts": _basics_contact_parts(basics),
    }

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
    return header, sections
