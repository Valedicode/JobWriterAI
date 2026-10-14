"""
Requirement rubric: every requirement of the posting, how the CV meets it,
and where on the CV it shows.

Built from the scorer's skill matches and eligibility checks plus the fact
ledger, without another LLM call:

- skill rows ('req.<n>'): must-haves, nice-to-haves, and skills the posting
  says it will teach ('learnable'), each with a status and the facts and
  bullets that evidence it;
- knockout rows ('ko.<n>'): languages, education and the posting's stated
  conditions (work mode, location, hours, start date, work permit). These are
  pass/fail and stay out of the weighted score.

``rubric_alignment`` turns the skill rows into one number that rewards a
requirement being shown in a bullet, not only listed in the skills section.
Re-running the evidence search on the tailored CV (``refresh_evidence``) and
comparing the two numbers is the before/after alignment delta.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from app.models.schemas import EligibilityCheck, RequirementRow, SkillMatch
from app.services import skill_graph
from app.services.fact_ledger import entities_in

STATUS_VALUE = {"met": 1.0, "partial": 0.6, "learnable_gap": 0.3, "missing": 0.0}
CATEGORY_WEIGHT = {"must_have": 2.0, "nice_to_have": 1.0}
# A requirement the CV meets but shows only in the skills list counts this
# much of one that a bullet demonstrates.
UNEVIDENCED_FACTOR = 0.7

_EVIDENCE_KINDS = {"bullet", "skill", "education"}

CONDITION_FIELDS = (
    ("work_mode", "Work mode"),
    ("location_constraint", "Location"),
    ("hours_per_week", "Hours per week"),
    ("start_date", "Start date"),
    ("work_permit", "Work permit"),
)

_BUCKET_STATUS = {"matched": "met", "transferable": "partial", "learnable": "learnable_gap", "missing": "missing"}
_ELIGIBILITY_STATUS = {"met": "met", "unmet": "missing", "unclear": "unclear"}


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def find_evidence(names: Iterable[str], ledger: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(fact ids, bullet ids) of the ledger facts that name any of ``names``:
    by skill-graph entity, or by a mention of the name itself (for skills the
    graph doesn't know)."""
    names = [n for n in dict.fromkeys(names) if n and str(n).strip()]
    canonical = {skill_graph.normalize(n) for n in names} - {""}
    canonical |= {e for n in names for e in entities_in(n)}
    patterns = [p for p in (skill_graph.mention_pattern(n) for n in names) if p is not None]

    fact_ids: list[str] = []
    bullet_ids: list[str] = []
    for fact_id, fact in (ledger or {}).items():
        if not isinstance(fact, dict) or fact.get("kind") not in _EVIDENCE_KINDS:
            continue
        value = str(fact.get("value") or "")
        if canonical & set(fact.get("entities") or []) or any(p.search(value) for p in patterns):
            fact_ids.append(fact_id)
            if fact.get("kind") == "bullet":
                bullet_ids.append(fact.get("bullet_id") or fact_id)
    return fact_ids, bullet_ids


def _evidence_names(text: str, alternatives: Sequence[str], matched_with: str | None) -> list[str]:
    return [*(alternatives or [text]), *([matched_with] if matched_with else [])]


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

def _groups_by_label(job_data: dict[str, Any]) -> dict[str, list[str]]:
    """OR-groups keyed by the label the scorer gives a collapsed group."""
    groups: dict[str, list[str]] = {}
    for raw in job_data.get("required_skill_groups") or []:
        if isinstance(raw, list):
            group = [str(s).strip() for s in raw if s and str(s).strip()]
            if len(group) > 1:
                groups[" / ".join(group)] = group
    return groups


def skill_rows(
    matches: Sequence[SkillMatch],
    buckets: Sequence[str],
    job_data: dict[str, Any],
    ledger: dict[str, dict[str, Any]],
    judge_model: str | None = None,
) -> list[RequirementRow]:
    """One row per skill match. ``buckets`` are the scorer's GapAnalysis
    buckets for the same matches (matched / transferable / learnable / missing)."""
    groups = _groups_by_label(job_data)
    taught = {_norm(s) for s in job_data.get("learnable_skills") or []}

    def category_of(match: SkillMatch) -> str:
        if match.is_required:
            return "must_have"
        return "learnable" if _norm(match.required_skill) in taught else "nice_to_have"

    # Must-haves first, then nice-to-haves, then skills the employer teaches;
    # the sort is stable, so ids stay the same for the same posting.
    order = {"must_have": 0, "nice_to_have": 1, "learnable": 2}
    ranked = sorted(zip(matches, buckets), key=lambda pair: order[category_of(pair[0])])

    rows: list[RequirementRow] = []
    for n, (match, bucket) in enumerate(ranked):
        status = _BUCKET_STATUS[bucket]
        category = category_of(match)
        if category == "learnable" and status == "missing":
            # The employer says it will teach this one: not having it yet is
            # a learnable gap, never a plain miss.
            status = "learnable_gap"
        alternatives = groups.get(match.required_skill, [])
        matched_with = match.matched_with if status != "missing" else None
        fact_ids, bullet_ids = find_evidence(
            _evidence_names(match.required_skill, alternatives, matched_with), ledger
        )
        scored_by = match.scored_by or ("skill_graph" if match.kind == "family" else "deterministic")
        rows.append(RequirementRow(
            requirement_id=f"req.{n}",
            text=match.required_skill,
            category=category,  # type: ignore[arg-type]
            alternatives=alternatives,
            status=status,  # type: ignore[arg-type]
            matched_with=matched_with,
            evidence_fact_ids=fact_ids,
            evidence_bullet_ids=bullet_ids,
            reason=match.rationale,
            scored_by=scored_by,
            judge_model=judge_model if scored_by == "llm_judge" else None,
        ))
    return rows


def knockout_rows(eligibility: Sequence[EligibilityCheck], job_data: dict[str, Any]) -> list[RequirementRow]:
    """Pass/fail rows: eligibility checks, then the posting's stated conditions,
    which no CV can show and the candidate has to check themselves."""
    rows = [
        RequirementRow(
            requirement_id=f"ko.{n}",
            text=check.requirement,
            category="knockout",
            status=_ELIGIBILITY_STATUS[check.status],  # type: ignore[arg-type]
            matched_with=check.matched_with,
            reason=check.rationale,
        )
        for n, check in enumerate(eligibility)
    ]
    for key, label in CONDITION_FIELDS:
        value = job_data.get(key)
        if isinstance(value, str) and value.strip():
            rows.append(RequirementRow(
                requirement_id=f"ko.{len(rows)}",
                text=f"{label}: {value.strip()}",
                category="knockout",
                status="unclear",
                reason="A CV can't show this; check that it works for you.",
            ))
    return rows


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def rubric_alignment(rows: Iterable[RequirementRow | dict[str, Any]]) -> float | None:
    """Weighted 0-1 fit of the must-have and nice-to-have rows (must-haves
    count double). A met requirement shown only in the skills list counts
    UNEVIDENCED_FACTOR of one a bullet demonstrates. None without such rows."""
    total = weighted = 0.0
    for raw in rows:
        row = raw if isinstance(raw, RequirementRow) else RequirementRow.model_validate(raw)
        weight = CATEGORY_WEIGHT.get(row.category)
        value = STATUS_VALUE.get(row.status)
        if weight is None or value is None:
            continue
        total += weight
        weighted += weight * value * (1.0 if row.evidence_bullet_ids else UNEVIDENCED_FACTOR)
    return round(weighted / total, 4) if total else None


def refresh_evidence(
    rows: Iterable[RequirementRow | dict[str, Any]], ledger: dict[str, dict[str, Any]]
) -> list[RequirementRow]:
    """The same rows with their evidence searched again in ``ledger`` (the
    tailored CV's). Statuses don't change: rewriting can't add a skill (the
    verifier blocks that), but it can show a held skill in a bullet."""
    refreshed = []
    for raw in rows:
        row = raw if isinstance(raw, RequirementRow) else RequirementRow.model_validate(raw)
        if row.category == "knockout":
            refreshed.append(row)
            continue
        fact_ids, bullet_ids = find_evidence(_evidence_names(row.text, row.alternatives, row.matched_with), ledger)
        refreshed.append(row.model_copy(update={"evidence_fact_ids": fact_ids, "evidence_bullet_ids": bullet_ids}))
    return refreshed


def evidence_prompt_block(rows: Iterable[dict[str, Any]], limit: int = 40) -> str:
    """The skill rows as prompt lines for bullet selection:
    `[req.0] must_have Python (met): experience.0.responsibilities.1, ...`."""
    lines = []
    for row in rows:
        if not isinstance(row, dict) or row.get("category") == "knockout":
            continue
        bullets = ", ".join(row.get("evidence_bullet_ids") or []) or "no bullet shows it"
        lines.append(f"[{row.get('requirement_id')}] {row.get('category')} {row.get('text')} "
                     f"({row.get('status')}): {bullets}")
    return "\n".join(lines[:limit]) if lines else "(none)"
