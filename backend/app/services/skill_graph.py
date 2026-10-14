"""
Skill graph service for compatibility scoring v2.

Provides alias normalization, family lookup, and intra-family transferability
scores from a curated JSON graph at backend/app/data/skill_graph.json.

The graph captures the insight that frameworks within the same family share
underlying paradigms (e.g. React/Vue/Angular are all component-based reactive
UI frameworks), so a candidate skilled in one is typically productive in
another within days or weeks.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "skill_graph.json"


@lru_cache(maxsize=1)
def load_skill_graph() -> Dict:
    """Load the skill graph JSON once and cache it."""
    if not _GRAPH_PATH.exists():
        return {"families": {}, "aliases": {}}
    with _GRAPH_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _member_to_family() -> Dict[str, str]:
    """Reverse-index every family member to its family name (normalized keys)."""
    graph = load_skill_graph()
    index: Dict[str, str] = {}
    for family_name, family in graph.get("families", {}).items():
        for member in family.get("members", []):
            index[_basic_normalize(member)] = family_name
    return index


@lru_cache(maxsize=1)
def _alias_map() -> Dict[str, str]:
    graph = load_skill_graph()
    return {_basic_normalize(k): _basic_normalize(v) for k, v in graph.get("aliases", {}).items()}


def _basic_normalize(skill: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation that doesn't disambiguate."""
    if skill is None:
        return ""
    s = str(skill).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


@lru_cache(maxsize=2048)
def normalize(skill: str) -> str:
    """
    Normalize a skill name for comparison.

    Lowercases, trims, collapses whitespace, then resolves through the alias
    map (so 'TF' -> 'tensorflow', 'k8s' -> 'kubernetes', etc.).
    """
    base = _basic_normalize(skill)
    if not base:
        return ""
    aliases = _alias_map()
    seen: Set[str] = set()
    current = base
    while current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


@lru_cache(maxsize=2048)
def find_family(skill: str) -> Optional[str]:
    """Return the family name a skill belongs to, or None if not found."""
    norm = normalize(skill)
    if not norm:
        return None
    return _member_to_family().get(norm)


def family_paradigm(family_name: str) -> Optional[str]:
    graph = load_skill_graph()
    fam = graph.get("families", {}).get(family_name)
    return fam.get("paradigm") if fam else None


def family_intra_transferability(family_name: str) -> Optional[float]:
    graph = load_skill_graph()
    fam = graph.get("families", {}).get(family_name)
    if not fam:
        return None
    val = fam.get("intra_transferability")
    return float(val) if val is not None else None


def is_direct_match(skill_a: str, skill_b: str) -> bool:
    """True if the two skill names are equivalent after alias normalization."""
    a = normalize(skill_a)
    b = normalize(skill_b)
    return bool(a) and a == b


def intra_family_transferability(skill_a: str, skill_b: str) -> Optional[float]:
    """
    Return the intra-family transferability score if both skills belong to the
    same family, else None. Returns None if it's a direct match (caller should
    handle that case explicitly with score 1.0).
    """
    if is_direct_match(skill_a, skill_b):
        return None
    fam_a = find_family(skill_a)
    fam_b = find_family(skill_b)
    if fam_a and fam_b and fam_a == fam_b:
        return family_intra_transferability(fam_a)
    return None


def best_family_match(
    required_skill: str, candidate_skills: List[str]
) -> Optional[Tuple[str, float, str]]:
    """
    Find the best candidate skill that shares a family with the required skill.

    Returns (matched_candidate_skill, transferability, family_name) for the
    first candidate found in the same family, else None. Direct matches are
    deliberately excluded (caller should detect those upstream).
    """
    fam = find_family(required_skill)
    if not fam:
        return None
    intra = family_intra_transferability(fam)
    if intra is None:
        return None
    for cand in candidate_skills:
        if is_direct_match(required_skill, cand):
            continue
        if find_family(cand) == fam:
            return (cand, intra, fam)
    return None


@lru_cache(maxsize=2048)
def surface_forms(skill: str) -> Tuple[str, ...]:
    """
    Every spelling a skill may appear under in free text: the raw normalized
    form, the alias-resolved canonical form, and every alias that resolves to
    that canonical form (so 'Node.js' also yields 'nodejs', 'node', 'node.js').
    """
    base = _basic_normalize(skill)
    if not base:
        return tuple()
    canonical = normalize(skill)
    forms = {base, canonical}
    for alias, target in _alias_map().items():
        if target == canonical:
            forms.add(alias)
    return tuple(sorted(forms, key=len, reverse=True))


@lru_cache(maxsize=2048)
def mention_pattern(skill: str) -> Optional["re.Pattern[str]"]:
    """
    Regex that finds any surface form of `skill` in free text.

    Uses lookarounds instead of \\b so names ending in a symbol ('C++', 'C#')
    or containing a dot ('Node.js') still match at their edges, while 'c'
    never matches inside 'c++' and 'java' never matches inside 'javascript'.
    """
    forms = surface_forms(skill)
    if not forms:
        return None
    alternation = "|".join(re.escape(f) for f in forms)
    return re.compile(rf"(?<![\w+#.])(?:{alternation})(?![\w+#])", re.IGNORECASE)


def list_known_skills() -> List[str]:
    """Flat list of every member skill across all families (normalized)."""
    return list(_member_to_family().keys())
