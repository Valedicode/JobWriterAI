"""
Rewrite verifier: deterministic checks of a rewrite pass against the fact
ledger, before the user reviews it.

Each rewritten bullet is compared with its original bullet and the facts it
cites (``cited_fact_ids``). A claim that no fact supports **blocks**:

- unknown_entity: names a technology the CV states nowhere;
- new_metric: states a number no fact has;
- changed_dates: states a year no fact has.

Weaker problems only **warn**, for the user to judge at the review gate:

- missing_citation: a technology or number the CV shows elsewhere (say, in the
  skills list) but that this bullet doesn't cite - true for the candidate,
  maybe not for this piece of work;
- banned_phrase: a CV cliché the original didn't use;
- length_blowup, structure_changed (no matching bullet);
- language_mismatch, when the language of a rewrite differs from its original's
  and the user hasn't chosen a CV language.

A rewrite in another language than the one the user chose for the CV **blocks**
(language_mismatch): a German posting must not turn an English CV German, so the
rewrite is run again in the chosen language, and the original wording is kept
if it still isn't.

The updated summary is a synthesis of the whole CV, so it is checked against
the whole ledger. Bullets the user wrote themselves are not checked.

No LLM is involved, so the same pass always gets the same verdict.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.models.schemas import VerifierIssue, VerifierResult
from app.services.fact_ledger import entities_in, metrics_in

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_THOUSANDS_RE = re.compile(r"^([$€£]?)(\d{1,3}(?:[,.]\d{3})+)(%|k|m|x|\+)?$")

# Phrases recruiters read as filler. Warned only when the rewrite adds them.
BANNED_PHRASES = (
    "results-driven", "results driven", "detail-oriented", "team player", "go-getter",
    "hard-working", "hardworking", "think outside the box", "synergy", "synergies",
    "best-in-class", "world-class", "rockstar", "ninja", "guru", "self-starter",
    "dynamic professional", "passionate about", "proven track record",
    "teamfähig", "belastbar", "zielorientiert", "hochmotiviert", "kommunikationsstark",
    "synergien", "lösungsorientiert",
)

_LANGUAGE_CODES = {"english": "en", "german": "de"}
_LANGUAGE_NAMES = {"en": "English", "de": "German"}

# Words that are the same in both languages ("in") would tip the balance, so
# each list has only words that belong to one of them.
_STOPWORDS = {
    "en": {"the", "and", "with", "for", "of", "to", "a", "on", "as", "was", "by", "from"},
    "de": {"der", "die", "das", "und", "mit", "für", "von", "zu", "im", "ein", "eine", "bei", "auf", "als", "durch"},
}


def text_language(text: str) -> str | None:
    """'en' or 'de' when ``text`` clearly reads as one, else None."""
    return _language(text)


def _language(text: str) -> str | None:
    words = re.findall(r"[a-zäöüß]+", text.casefold())
    counts = {lang: sum(w in stop for w in words) for lang, stop in _STOPWORDS.items()}
    for lang, other in (("en", "de"), ("de", "en")):
        if counts[lang] >= 2 and counts[lang] > 2 * counts[other]:
            return lang
    return None


def normalize_metric(value: str) -> str:
    """'1,200' == '1.200' == '1200'; '30 %' == '30 percent' == '30%'."""
    text = value.casefold().replace(" ", "").replace("percent", "%").replace("prozent", "%")
    match = _THOUSANDS_RE.match(text)
    if match:
        text = f"{match.group(1)}{re.sub(r'[,.]', '', match.group(2))}{match.group(3) or ''}"
    return text


_NUMBER_WORDS = {
    word: str(n)
    for words, n in (
        (("one", "ein", "eine", "einen"), 1), (("two", "zwei"), 2), (("three", "drei"), 3), (("four", "vier"), 4),
        (("five", "fünf"), 5), (("six", "sechs"), 6), (("seven", "sieben"), 7), (("eight", "acht"), 8),
        (("nine", "neun"), 9), (("ten", "zehn"), 10), (("eleven", "elf"), 11), (("twelve", "zwölf"), 12),
    )
    for word in words
}


def numbers_in(texts: Iterable[str]) -> set[str]:
    """Numbers in ``texts``, normalised. Number words count as their digits,
    so rewriting 'two teams' as '2 teams' adds nothing."""
    return _metric_set(texts)


def _metric_set(texts: Iterable[str]) -> set[str]:
    """See ``numbers_in``."""
    found: set[str] = set()
    for text in texts:
        found.update(normalize_metric(m) for m in metrics_in(text))
        found.update(_NUMBER_WORDS[w] for w in re.findall(r"[a-zäöüß]+", text.casefold()) if w in _NUMBER_WORDS)
    return found


def _fact_texts(facts: Iterable[dict[str, Any]]) -> list[str]:
    return [t for f in facts for t in (f.get("value", ""), f.get("source_text", "")) if isinstance(t, str)]


class _Ledger:
    """Lookups over the fact ledger used by every check."""

    def __init__(self, ledger: dict[str, dict[str, Any]]):
        self.facts = ledger or {}
        self.all_texts = _fact_texts(self.facts.values())
        self.all_entities = {e for f in self.facts.values() for e in f.get("entities") or []}
        self.all_metrics = _metric_set(self.all_texts)
        self.all_years = {y for t in self.all_texts for y in _YEAR_RE.findall(t)}

    def where(self, entity: str) -> list[str]:
        return [fid for fid, f in self.facts.items() if entity in (f.get("entities") or [])]

    def own_facts(self, bullet_id: str | None) -> list[dict[str, Any]]:
        if not bullet_id:
            return []
        return [f for fid, f in self.facts.items() if fid == bullet_id or fid.startswith(f"{bullet_id}.metric.")]


def _check_text(
    *,
    bullet_id: str | None,
    original: str,
    rewritten: str,
    support: list[dict[str, Any]],
    ledger: _Ledger,
    whole_ledger_supports: bool,
    cv_language: str | None = None,
) -> list[VerifierIssue]:
    """Claim checks shared by bullets and the summary. With
    ``whole_ledger_supports`` (the summary), any fact counts as support.
    ``cv_language`` ("english" / "german") is the language the user chose."""
    issues: list[VerifierIssue] = []
    support_texts = [original, *_fact_texts(support)]

    def issue(check: str, severity: str, detail: str, span: str | None = None) -> None:
        issues.append(VerifierIssue(bullet_id=bullet_id, check=check, severity=severity,  # type: ignore[arg-type]
                                    detail=detail, offending_span=span))

    supported_entities = set(entities_in(original)) | {e for f in support for e in f.get("entities") or []}
    for entity in entities_in(rewritten):
        if entity in supported_entities or (whole_ledger_supports and entity in ledger.all_entities):
            continue
        elsewhere = ledger.where(entity)
        if elsewhere:
            issue("missing_citation", "warn",
                  f"Mentions {entity}, which your CV shows elsewhere ({', '.join(elsewhere[:3])}) "
                  "but not for this bullet. Check it's true here.", entity)
        else:
            issue("unknown_entity", "block", f"Mentions {entity}, which nothing on your CV supports.", entity)

    supported_metrics = _metric_set(support_texts)
    for metric in metrics_in(rewritten):
        norm = normalize_metric(metric)
        if norm in supported_metrics or (whole_ledger_supports and norm in ledger.all_metrics):
            continue
        if norm in ledger.all_metrics:
            issue("missing_citation", "warn",
                  f"States {metric}, a number your CV uses elsewhere, not for this bullet.", metric)
        else:
            issue("new_metric", "block", f"States {metric}, a number that isn't on your CV.", metric)

    supported_years = {y for t in support_texts for y in _YEAR_RE.findall(t)}
    for year in dict.fromkeys(_YEAR_RE.findall(rewritten)):
        if year in supported_years or (whole_ledger_supports and year in ledger.all_years):
            continue
        issue("changed_dates", "block", f"States the year {year}, which the original doesn't.", year)

    lowered_original, lowered = original.casefold(), rewritten.casefold()
    for phrase in BANNED_PHRASES:
        if phrase in lowered and phrase not in lowered_original:
            issue("banned_phrase", "warn", f"Adds the filler phrase '{phrase}'.", phrase)

    if len(rewritten) > 160 and len(rewritten) > 2 * max(len(original), 1):
        issue("length_blowup", "warn", f"Grew from {len(original)} to {len(rewritten)} characters.")

    before, after = _language(original), _language(rewritten)
    chosen = _LANGUAGE_CODES.get(cv_language or "")
    if chosen:
        if after and after != chosen:
            issue("language_mismatch", "block",
                  f"Is written in {_LANGUAGE_NAMES[after]}, but your CV is in {_LANGUAGE_NAMES[chosen]}.")
    elif before and after and before != after:
        issue("language_mismatch", "warn", f"Switches the language from {before} to {after}.")
    return issues


def _preservation(original: str, rewritten: str) -> float | None:
    """Share of the original's technologies and numbers the rewrite kept,
    or None when the original names none."""
    entities = set(entities_in(original))
    metrics = _metric_set([original])
    total = len(entities) + len(metrics)
    if not total:
        return None
    kept = len(entities & set(entities_in(rewritten))) + len(metrics & _metric_set([rewritten]))
    return kept / total


def verify_rewrites(
    rewritten_content: dict[str, Any],
    fact_ledger: dict[str, dict[str, Any]],
    bullet_index: dict[str, dict[str, Any]],
    *,
    cv_summary: str | None = None,
    attempt: int = 1,
    user_written_ids: Iterable[str] = (),
    cv_language: str | None = None,
) -> VerifierResult:
    ledger = _Ledger(fact_ledger)
    skip = set(user_written_ids)
    issues: list[VerifierIssue] = []
    preservation: list[float] = []

    for bullet in rewritten_content.get("rewritten_bullets") or []:
        if not isinstance(bullet, dict):
            continue
        bullet_id = bullet.get("bullet_id")
        rewritten = (bullet.get("rewritten") or "").strip()
        ref = bullet_index.get(bullet_id) if isinstance(bullet_id, str) else None
        original = (ref or {}).get("text") or bullet.get("original") or ""
        if bullet_id in skip or not rewritten or rewritten == original.strip():
            continue
        if ref is None:
            issues.append(VerifierIssue(
                bullet_id=bullet_id, check="structure_changed", severity="warn",
                detail="This rewrite doesn't match a bullet on your CV, so it can't be placed.",
            ))

        cited = [c for c in bullet.get("cited_fact_ids") or [] if isinstance(c, str)]
        unknown = [c for c in cited if c not in ledger.facts]
        if unknown:
            issues.append(VerifierIssue(
                bullet_id=bullet_id, check="missing_citation", severity="warn",
                detail=f"Cites facts that don't exist: {', '.join(unknown[:3])}.",
            ))
        support = ledger.own_facts(bullet_id) + [ledger.facts[c] for c in cited if c in ledger.facts]
        issues.extend(_check_text(
            bullet_id=bullet_id, original=original, rewritten=rewritten,
            support=support, ledger=ledger, whole_ledger_supports=False, cv_language=cv_language,
        ))
        score = _preservation(original, rewritten)
        if score is not None:
            preservation.append(score)

    summary = (rewritten_content.get("updated_summary") or "").strip()
    if summary and summary != (cv_summary or "").strip() and "summary" not in skip:
        issues.extend(_check_text(
            bullet_id="summary", original=cv_summary or "", rewritten=summary,
            support=[], ledger=ledger, whole_ledger_supports=True, cv_language=cv_language,
        ))

    return VerifierResult(
        passed=not any(i.severity == "block" for i in issues),
        attempt=attempt,
        issues=issues,
        preservation_score=round(sum(preservation) / len(preservation), 3) if preservation else 1.0,
    )


def retry_feedback(result: VerifierResult) -> str:
    """The blocking problems, phrased as instructions for the next rewrite."""
    lines = [
        f"Bullet {issue.bullet_id}: {issue.detail} "
        + ("Write it again in the CV's language." if issue.check == "language_mismatch"
           else "Remove it or reword without it.")
        for issue in result.issues if issue.severity == "block"
    ]
    return "The fact check rejected these rewrites. " + " ".join(lines) if lines else ""


def issue_notes(result: VerifierResult) -> dict[str, list[str]]:
    """Issues grouped by bullet_id, as short lines for the review gate."""
    notes: dict[str, list[str]] = {}
    for issue in result.issues:
        notes.setdefault(issue.bullet_id or "", []).append(
            f"{'Blocked' if issue.severity == 'block' else 'Check'}: {issue.detail}"
        )
    return notes
