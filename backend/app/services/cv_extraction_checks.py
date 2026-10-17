"""
Deterministic checks on a CV the extraction model returned, against the PDF's own text.

The model reads the PDF's text and fills ResumeInfo. It sometimes slips in ways the text
itself contradicts, and the same CV can come out right in one run and wrong in the next:

- a bullet copied into a second entry, e.g. an activity with only a paragraph gets the
  bullets of the jobs that follow it;
- the grade repeated in an education entry's details ("Current Grade: 1.6" next to the
  grade itself);
- a job without an employer (private tutoring) given the job title as its company, so
  the title prints twice.

Each check only removes what the PDF text shows is a copy or an echo; it never moves or
invents content. Every change is returned as a note for the log.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

# (section, bullet list field, fields naming the entry, most specific first)
_BULLET_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("experience", "responsibilities", ("position", "company")),
    ("leadership_activities", "highlights", ("role", "organization")),
    ("projects", "outcomes", ("name",)),
)
# Two bullets this similar (after normalising) are the same bullet.
DUPLICATE_SIMILARITY = 0.85
_KEY_CHARS = 48
_MIN_KEY_CHARS = 16
_GRADE_WORDS = re.compile(
    r"\b(grade|gpa|note|notendurchschnitt|durchschnittsnote|durchschnitt|score|cgpa)\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _normalize(text: str) -> str:
    """Lowercase letters and digits separated by single spaces; PDF line-break
    hyphens joined, ligatures unfolded."""
    text = unicodedata.normalize("NFKC", text or "").replace("-\n", "")
    return re.sub(r"[^0-9a-zäöüß]+", " ", text.casefold()).strip()


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(text or "")}


def _similar(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= DUPLICATE_SIMILARITY


def repeats_grade(detail: str, grade: str) -> bool:
    """Whether an education detail only restates the grade ("Current Grade: 1.6" next to "1.6")."""
    grade_numbers = _numbers(grade)
    return bool(grade_numbers) and bool(_GRADE_WORDS.search(detail or "")) and _numbers(detail) == grade_numbers


def same_text(a: str, b: str) -> bool:
    """Equal after normalising case, punctuation and dashes ("Tutor – Maths" == "tutor - maths")."""
    a, b = _normalize(a), _normalize(b)
    return bool(a) and a == b


def drop_grade_echoes(cv: dict[str, Any]) -> list[str]:
    """Remove education details that only repeat the entry's grade."""
    notes: list[str] = []
    for entry in cv.get("education") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("details"), list):
            continue
        grade = str(entry.get("grade") or "")
        kept = []
        for detail in entry["details"]:
            if repeats_grade(str(detail), grade):
                notes.append(f"dropped a detail repeating the grade: {detail!r}")
            else:
                kept.append(detail)
        entry["details"] = kept
    return notes


def clear_company_echoes(cv: dict[str, Any]) -> list[str]:
    """A company equal to the position is the title read twice, not an employer."""
    notes: list[str] = []
    for entry in cv.get("experience") or []:
        if not isinstance(entry, dict):
            continue
        if same_text(str(entry.get("company") or ""), str(entry.get("position") or "")):
            notes.append(f"cleared a company repeating the position: {entry.get('company')!r}")
            entry["company"] = None
    return notes


def _bullet_refs(cv: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for section, field, name_fields in _BULLET_SECTIONS:
        for i, entry in enumerate(cv.get(section) or []):
            if not isinstance(entry, dict):
                continue
            names = [str(entry.get(f) or "") for f in name_fields]
            for j, bullet in enumerate(entry.get(field) or []):
                if isinstance(bullet, str) and len(_normalize(bullet)) >= _MIN_KEY_CHARS:
                    refs.append({"section": section, "field": field, "entry": i, "idx": j,
                                 "text": bullet, "norm": _normalize(bullet), "names": names})
    return refs


class _Source:
    """The PDF text normalised, joined into one string, with the offset where
    each line starts (entry names are matched only at the start of a line, so
    a project mentioned inside another entry's text isn't taken for its header)."""

    def __init__(self, text: str) -> None:
        text = unicodedata.normalize("NFKC", text or "").replace("-\n", "")
        self.lines: list[tuple[int, str]] = []
        parts, offset = [], 0
        for raw in text.splitlines():
            line = _normalize(raw)
            if line:
                self.lines.append((offset, line))
                parts.append(line)
                offset += len(line) + 1
        self.joined = " ".join(parts)

    def positions(self, needle: str) -> list[int]:
        """Start offsets of every occurrence of the needle's opening words."""
        key = needle[:_KEY_CHARS]
        if len(key) < _MIN_KEY_CHARS:
            return []
        found, start = [], 0
        while (at := self.joined.find(key, start)) != -1:
            found.append(at)
            start = at + 1
        return found

    def header_before(self, names: list[str], at: int) -> int:
        """Start of the last line before ``at`` that begins with one of the
        entry's names (-1 if there is none)."""
        keys = [k for k in (_normalize(n) for n in names) if len(k) >= 3]
        best = -1
        for offset, line in self.lines:
            if offset >= at:
                break
            if any(line.startswith(k) for k in keys):
                best = offset
        return best


def drop_cross_entry_duplicates(cv: dict[str, Any], source_text: str) -> list[str]:
    """
    Keep each bullet only under the entry it appears under in the PDF.

    When the same bullet sits in two entries and the PDF text has it once, it
    belongs to the entry whose name comes closest before it in the text; the
    other copies are removed. A bullet the PDF itself repeats, or that can't be
    located, is left alone.
    """
    source = _Source(source_text)
    refs = _bullet_refs(cv)
    groups: list[list[dict[str, Any]]] = []
    for ref in refs:
        for group in groups:
            if _similar(group[0]["norm"], ref["norm"]):
                group.append(ref)
                break
        else:
            groups.append([ref])

    notes: list[str] = []
    to_remove: set[tuple[str, str, int, int]] = set()
    for group in groups:
        entries = {(r["section"], r["entry"]) for r in group}
        if len(entries) < 2:
            continue
        found = source.positions(group[0]["norm"]) or source.positions(group[0]["norm"][-_KEY_CHARS:])
        if len(found) != 1:
            notes.append(f"left a repeated bullet in place (found {len(found)} times in the PDF): {group[0]['text'][:60]!r}")
            continue
        owner = max(group, key=lambda r: source.header_before(r["names"], found[0]))
        if source.header_before(owner["names"], found[0]) < 0:
            notes.append(f"left a repeated bullet in place (no entry name before it): {group[0]['text'][:60]!r}")
            continue
        for ref in group:
            if (ref["section"], ref["entry"]) != (owner["section"], owner["entry"]):
                to_remove.add((ref["section"], ref["field"], ref["entry"], ref["idx"]))
                notes.append(
                    f"removed a bullet copied into {ref['section']}[{ref['entry']}] "
                    f"(it belongs to {owner['section']}[{owner['entry']}]): {ref['text'][:60]!r}"
                )

    for section, field, entry_idx, idx in sorted(to_remove, key=lambda k: -k[3]):
        del cv[section][entry_idx][field][idx]
    return notes


def clean_extracted_cv(cv: dict[str, Any], source_text: str) -> tuple[dict[str, Any], list[str]]:
    """The CV with extraction copies and echoes removed, and a note per change."""
    cleaned = copy.deepcopy(cv)
    notes = drop_cross_entry_duplicates(cleaned, source_text)
    notes += drop_grade_echoes(cleaned)
    notes += clear_company_echoes(cleaned)
    return cleaned, notes
