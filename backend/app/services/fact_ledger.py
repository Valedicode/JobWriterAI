"""
Fact ledger: everything the candidate's CV states, as addressable facts.

Built deterministically (no LLM) from the extracted CV and its bullet index
when a job-tailoring run starts, then extended with what the user confirms
during the session (eligibility answers, profile additions in the chat). A
rewrite may only claim what some fact supports; the verifier (plan item 10)
checks rewrites against this ledger.

Each fact carries the technologies it names (``entities``, in
``skill_graph.normalize`` form) so "this rewrite mentions Kubernetes" can be
checked against "some fact names Kubernetes" without fuzzy matching.

Fact ids:
- bullets reuse their bullet_id ("experience.0.responsibilities.1", "summary");
- numbers in a bullet: "<bullet_id>.metric.<n>";
- entry facts: "experience.0.employer", "experience.0.title", "experience.0.dates",
  the same for "leadership_activities.<i>"; "projects.<i>.title";
  "education.<i>", "education.<i>.dates", "education.<i>.grade", "education.<i>.details.<j>";
- "skill.<slug>", "lang.<slug>";
- user answers: "user.<slug>".
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable

from app.models.schemas import FactEntry
from app.services import skill_graph

# Short skill names that are also ordinary words or letters ("go", "c", "es",
# "ts") only count when written like a name: "Go", "GO", "C", "TS".
_SHORT_FORM_MAX = 2

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_METRIC_RE = re.compile(
    r"(?<![\w.])(?:[$€£]\s?)?\d+(?:[.,]\d+)*(?:\s?(?:%|percent|prozent|k|m|x|\+))?(?!\w)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Entities and numbers in free text
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _entity_patterns() -> tuple[re.Pattern[str] | None, re.Pattern[str] | None, dict[str, str]]:
    """(case-insensitive pattern for long forms, case-sensitive pattern for
    short forms, surface form -> canonical skill)."""
    canonical_by_form: dict[str, str] = {}
    for skill in skill_graph.list_known_skills():
        for form in skill_graph.surface_forms(skill):
            canonical_by_form.setdefault(form, skill_graph.normalize(skill))
    for alias in skill_graph.load_skill_graph().get("aliases", {}):
        for form in skill_graph.surface_forms(alias):
            canonical_by_form.setdefault(form, skill_graph.normalize(alias))

    def compile_forms(forms: Iterable[str], flags: int) -> re.Pattern[str] | None:
        ordered = sorted(set(forms), key=len, reverse=True)
        if not ordered:
            return None
        alternation = "|".join(re.escape(f) for f in ordered)
        # Skill names contain '+', '#' and '.', so \b isn't usable.
        return re.compile(rf"(?<![\w+#])(?:{alternation})(?![\w+#])", flags)

    long_forms = [f for f in canonical_by_form if len(f) > _SHORT_FORM_MAX]
    short_forms = [
        variant
        for f in canonical_by_form if len(f) <= _SHORT_FORM_MAX
        for variant in {f.upper(), f.capitalize()}
    ]
    return (
        compile_forms(long_forms, re.IGNORECASE),
        compile_forms(short_forms, 0),
        canonical_by_form,
    )


def entities_in(text: str) -> list[str]:
    """Known technologies named in ``text``, canonical and de-duplicated, in
    order of first appearance."""
    if not isinstance(text, str) or not text.strip():
        return []
    long_re, short_re, canonical_by_form = _entity_patterns()
    found: list[tuple[int, str]] = []
    for pattern in (long_re, short_re):
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            canonical = canonical_by_form.get(match.group(0).lower())
            if canonical:
                found.append((match.start(), canonical))
    return list(dict.fromkeys(name for _, name in sorted(found)))


def metrics_in(text: str) -> list[str]:
    """Numbers stated in ``text`` (counts, percentages, amounts), not years."""
    if not isinstance(text, str):
        return []
    values = []
    for match in _METRIC_RE.finditer(text):
        value = match.group(0).strip()
        if _YEAR_RE.match(value):
            continue
        values.append(value)
    return list(dict.fromkeys(values))


def _slug(value: str) -> str:
    text = skill_graph.normalize(value) or value.casefold()
    text = text.replace("#", "sharp").replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "item"


# --------------------------------------------------------------------------- #
# Building the ledger from the CV
# --------------------------------------------------------------------------- #

def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _fact(
    fact_id: str,
    kind: str,
    value: str,
    source_text: str | None = None,
    *,
    entities: list[str] | None = None,
    source: str = "cv_extraction",
    **extra: Any,
) -> FactEntry:
    """``entities=None`` means: find them in ``value``."""
    return FactEntry(
        fact_id=fact_id,
        kind=kind,  # type: ignore[arg-type]
        value=value,
        source=source,  # type: ignore[arg-type]
        source_text=source_text if source_text is not None else value,
        entities=entities_in(value) if entities is None else entities,
        **extra,
    )


def _skill_fact(name: str, source_text: str, **extra: Any) -> FactEntry:
    canonical = skill_graph.normalize(name)
    entities = list(dict.fromkeys(([canonical] if canonical else []) + entities_in(name)))
    return _fact(f"skill.{_slug(name)}", "skill", name, source_text, entities=entities, **extra)


def _language_value(language: str, fluency: str | None) -> str:
    return f"{language} ({fluency})" if fluency else language


def build_fact_ledger(cv_data: dict[str, Any], bullet_index: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{fact_id: FactEntry dump} for everything the CV states."""
    facts: list[FactEntry] = []
    cv = cv_data if isinstance(cv_data, dict) else {}

    # Skills section first: a skill also listed under a project keeps the
    # skills section as its source.
    for skill in cv.get("skills") or []:
        if _text(skill):
            facts.append(_skill_fact(_text(skill), _text(skill)))

    for entry in cv.get("languages") or []:
        if isinstance(entry, dict) and _text(entry.get("language")):
            value = _language_value(_text(entry["language"]), _text(entry.get("fluency")) or None)
            facts.append(_fact(f"lang.{_slug(entry['language'])}", "language", value, entities=[]))

    for bullet_id, ref in bullet_index.items():
        text = _text(ref.get("text"))
        if not text:
            continue
        facts.append(_fact(bullet_id, "bullet", text, bullet_id=bullet_id))
        for n, metric in enumerate(metrics_in(text)):
            facts.append(_fact(f"{bullet_id}.metric.{n}", "metric", metric, text, bullet_id=bullet_id, entities=[]))

    for section in ("experience", "leadership_activities"):
        title_key, employer_key, dates_key = (
            ("position", "company", "duration") if section == "experience" else ("role", "organization", "dates")
        )
        for i, entry in enumerate(cv.get(section) or []):
            if not isinstance(entry, dict):
                continue
            for suffix, kind, key in (("title", "title", title_key), ("employer", "employer", employer_key),
                                      ("dates", "date_range", dates_key)):
                value = _text(entry.get(key))
                if value:
                    facts.append(_fact(f"{section}.{i}.{suffix}", kind, value,
                                       entities=[] if kind == "date_range" else None))

    for i, entry in enumerate(cv.get("projects") or []):
        if not isinstance(entry, dict):
            continue
        name = _text(entry.get("name"))
        if name:
            facts.append(_fact(f"projects.{i}.title", "title", name))
        for tech in entry.get("technologies") or []:
            if _text(tech):
                facts.append(_skill_fact(_text(tech), f"Project {name or i + 1}: {_text(tech)}"))

    for i, entry in enumerate(cv.get("education") or []):
        if isinstance(entry, str) and entry.strip():
            facts.append(_fact(f"education.{i}", "education", entry.strip()))
            continue
        if not isinstance(entry, dict):
            continue
        heading = ", ".join(v for v in (_text(entry.get("degree")), _text(entry.get("institution"))) if v)
        if heading:
            facts.append(_fact(f"education.{i}", "education", heading))
        if _text(entry.get("dates")):
            facts.append(_fact(f"education.{i}.dates", "date_range", _text(entry["dates"]), entities=[]))
        if _text(entry.get("grade")):
            facts.append(_fact(f"education.{i}.grade", "metric", _text(entry["grade"]), entities=[]))
        for j, detail in enumerate(entry.get("details") or []):
            if _text(detail):
                facts.append(_fact(f"education.{i}.details.{j}", "education", _text(detail)))

    ledger: dict[str, dict[str, Any]] = {}
    for fact in facts:
        # A skill listed twice (skills section and a project) keeps its first source.
        ledger.setdefault(fact.fact_id, fact.model_dump())
    return ledger


# --------------------------------------------------------------------------- #
# Facts the user confirms during the session
# --------------------------------------------------------------------------- #

_NEGATIVE = {"no", "none", "nein", "n/a", "not applicable", "keine"}


def eligibility_answer_facts(questions: list[dict[str, Any]], answers: dict[str, str]) -> list[dict[str, Any]]:
    """Facts for the answers given at the confirm_eligibility gate. A 'no'
    is a fact too: the writer must not suggest the opposite."""
    facts: list[FactEntry] = []
    for q in questions:
        answer = _text(answers.get(q.get("id", "")))
        if not answer:
            continue
        source_text = f"{q.get('prompt', q.get('requirement', ''))} {answer}".strip()
        negative = answer.casefold() in _NEGATIVE
        if q.get("kind") == "language" and q.get("language"):
            language = q["language"]
            if negative:
                facts.append(_fact(f"user.{_slug(language)}", "user_stated", f"Does not speak {language}",
                                   source_text, entities=[], source="user_confirmed", user_verified=True))
            else:
                facts.append(_fact(f"lang.{_slug(language)}", "language", _language_value(language, answer),
                                   source_text, entities=[], source="user_confirmed", user_verified=True))
            continue
        requirement = _text(q.get("requirement"))
        value = f"Does not meet: {requirement}" if negative else answer
        facts.append(_fact(f"user.{_slug(requirement)}", "user_stated" if negative else "education", value,
                           source_text, source="user_confirmed", user_verified=True))
    return [f.model_dump() for f in facts]


def user_added_fact(kind: str, value: str, fluency: str | None, source_text: str) -> dict[str, Any]:
    """A skill or language the user added through the chat and confirmed."""
    if kind == "language":
        fact = _fact(f"lang.{_slug(value)}", "language", _language_value(value, fluency), source_text,
                     entities=[], source="user_added", user_verified=True)
    else:
        fact = _skill_fact(value, source_text, source="user_added", user_verified=True)
    return fact.model_dump()


def with_facts(ledger: dict[str, dict[str, Any]] | None, facts: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """A copy of ``ledger`` with ``facts`` added; a fact with the same id is replaced."""
    updated = dict(ledger or {})
    for fact in facts:
        updated[fact["fact_id"]] = fact
    return updated
