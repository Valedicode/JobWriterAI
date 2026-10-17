"""
Stable ids for every bullet-like string in a CV.

Every bullet gets an id of the form "<section>.<entry_idx>.<field>.<idx>"
(or "<section>.<entry_idx>.<field>" for a single text field, and "summary"
for the top-level summary). The id survives selection -> rewrite -> merge,
and the fact ledger, the verifier and the requirement rubric address bullets
by it.

Field coverage mirrors what ``resume_adapter.build_harvard_sections`` (the
LaTeX renderer's input) actually reads, so every id here lands somewhere in
the generated PDF.
"""

from __future__ import annotations

from typing import Any

_BULLET_FIELDS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "experience": ("responsibilities",),
    "projects": ("outcomes",),
    "leadership_activities": ("highlights",),
}
_TEXT_FIELDS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "projects": ("description",),
    "leadership_activities": ("description",),
}


def index_cv_bullets(cv_data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """
    Build {bullet_id: {section, entry_idx, field, idx, text}} for every
    bullet/description string in the CV, plus the top-level summary.
    """
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(cv_data, dict):
        return index

    for section, fields in _BULLET_FIELDS_BY_SECTION.items():
        entries = cv_data.get(section)
        if not isinstance(entries, list):
            continue
        for entry_idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            for field in fields:
                values = entry.get(field)
                if not isinstance(values, list):
                    continue
                for idx, text in enumerate(values):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    bullet_id = f"{section}.{entry_idx}.{field}.{idx}"
                    index[bullet_id] = {
                        "section": section,
                        "entry_idx": entry_idx,
                        "field": field,
                        "idx": idx,
                        "text": text,
                    }
            for field in _TEXT_FIELDS_BY_SECTION.get(section, ()):
                text = entry.get(field)
                if isinstance(text, str) and text.strip():
                    bullet_id = f"{section}.{entry_idx}.{field}"
                    index[bullet_id] = {
                        "section": section,
                        "entry_idx": entry_idx,
                        "field": field,
                        "idx": None,
                        "text": text,
                    }

    summary = cv_data.get("summary")
    if isinstance(summary, str) and summary.strip():
        index["summary"] = {"section": "summary", "entry_idx": None, "field": "summary", "idx": None, "text": summary}

    return index
