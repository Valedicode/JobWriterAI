"""
Scoring Agent — Compatibility Scoring and Skill-Matching Tools

Standalone module extracted from writer_agent.py. Contains all
compatibility-scoring logic that can be invoked independently of the Writer's
chat workflow:

  - Utility helpers (embeddings, BM25, cosine similarity)
  - Compatibility v2 constants and scoring weights
  - Skill-matching cascade (direct → family → embedding → LLM)
  - Multi-dimensional compatibility scorer (5 dimensions)
  - Gap analysis (matched / transferable / missing / over-qualified)
  - Seven public @tool functions registered by writer_agent in the LangChain agent

The Writer agent imports the @tool functions and _calculate_compatibility_score_v2_internal
for use in decide_tailoring_strategy's fallback path. All other consumers can
import directly from this module.

No logic has been changed relative to writer_agent.py — this is a pure structural
extraction.
"""

from __future__ import annotations

from pydantic import Field, BaseModel
from langchain_core.tools import tool
from app.openai_llm import chat_openai_mini, model_id, openai_client
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Literal, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

from app.models.schemas import (
    SkillMatch,
    DimensionScore,
    GapAnalysis,
    CompatibilityReport,
    EligibilityCheck,
)
from app.services import skill_graph as skill_graph_service


# ============================================
# OpenAI Client
# ============================================

def _get_openai_client() -> OpenAI:
    """Get the OpenAI-compatible client for embeddings (OpenRouter or OpenAI)."""
    return openai_client()


# ============================================
# Utility Helpers — Embeddings, BM25, Cosine
# ============================================

def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into words for BM25."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Get embeddings for a list of texts using OpenAI.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model (default: text-embedding-3-small)

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    client = _get_openai_client()

    embeddings = []
    batch_size = 100  # OpenAI allows up to 2048 inputs per request

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model_id(model),
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


@lru_cache(maxsize=4096)
def _get_single_embedding_cached(text: str, model: str = "text-embedding-3-small") -> Tuple[float, ...]:
    """
    Cached single-text embedding lookup.

    Skill names recur heavily across scoring runs (every job posting mentions
    'python', 'react', 'aws'), so caching the per-skill embedding cuts both
    cost and latency. Returns a tuple so the value is hashable; callers should
    convert to list as needed.
    """
    if not text or not text.strip():
        return tuple()
    embedding = get_embeddings([text], model=model)[0]
    return tuple(embedding)


def get_skill_embedding(skill: str, model: str = "text-embedding-3-small") -> List[float]:
    """Convenience wrapper around the cached single-embedding lookup."""
    return list(_get_single_embedding_cached(skill, model=model))


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return max(0.0, min(1.0, similarity))


def _calculate_bm25_scores(query_text: str, document_texts: List[str]) -> List[float]:
    """
    Calculate BM25 scores for query against multiple documents.

    Args:
        query_text: Query text to search for
        document_texts: List of document texts to search in

    Returns:
        List of BM25 scores (one per document)
    """
    tokenized_docs = [_tokenize_text(doc) for doc in document_texts]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = _tokenize_text(query_text)
    scores = bm25.get_scores(tokenized_query)

    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.5] * len(scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores


# ============================================
# Public Scoring Tools — Similarity / BM25
# ============================================

@tool
def calculate_semantic_similarity(resume_text: str, job_text: str) -> str:
    """
    Calculate semantic similarity between resume content and job requirements using cosine similarity of embeddings.

    Args:
        resume_text: Text content from resume (can be full resume or specific section)
        job_text: Text content from job posting (can be full job description or specific requirements)

    Returns:
        JSON string with similarity score (0-1) and breakdown
    """
    try:
        resume_embedding = get_embeddings([resume_text])[0]
        job_embedding = get_embeddings([job_text])[0]
        similarity_score = cosine_similarity(resume_embedding, job_embedding)
        return json.dumps({
            "similarity_score": round(similarity_score, 4),
            "method": "cosine_similarity",
            "embedding_model": "text-embedding-3-small"
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "similarity_score": 0.0
        })


@tool
def calculate_bm25_score(resume_sections: str, job_requirements: str) -> str:
    """
    Calculate BM25 scores for resume sections against job requirements.

    Args:
        resume_sections: JSON string with resume sections (list of section texts)
        job_requirements: Job requirements text to match against

    Returns:
        JSON string with BM25 scores per section
    """
    try:
        sections_data = json.loads(resume_sections)

        if isinstance(sections_data, list):
            section_texts = sections_data
        elif isinstance(sections_data, dict):
            section_texts = []
            for key, value in sections_data.items():
                if isinstance(value, str):
                    section_texts.append(value)
                elif isinstance(value, list):
                    section_texts.extend([str(v) for v in value if isinstance(v, str)])
        else:
            section_texts = [str(sections_data)]

        scores = _calculate_bm25_scores(job_requirements, section_texts)

        result = {
            "bm25_scores": [round(score, 4) for score in scores],
            "average_score": round(sum(scores) / len(scores) if scores else 0.0, 4),
            "method": "BM25"
        }

        if isinstance(sections_data, dict):
            result["section_breakdown"] = {
                key: round(score, 4)
                for key, score in zip(sections_data.keys(), scores[:len(sections_data)])
            }

        return json.dumps(result)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "bm25_scores": [],
            "average_score": 0.0
        })


# ============================================
# Compatibility Scoring v2 — Constants
# ============================================
#
# Default weights for the five dimensions. Tweakable without code changes by
# editing this dict; downstream tools read from it. Sum should be 1.0.

COMPATIBILITY_WEIGHTS: Dict[str, float] = {
    "hard_skills": 0.40,
    "eligibility": 0.15,   # languages / degree / enrollment; omitted when the posting states none
    "experience": 0.15,
    "seniority": 0.10,
    "domain": 0.10,
    "ats_keywords": 0.10,
}

COMPATIBILITY_BANDS = [
    (0.45, "low"),
    (0.65, "medium"),
    (0.80, "high"),
    (1.01, "excellent"),
]

# Embedding cosine threshold below which we kick a skill pair out to the LLM
EMBEDDING_TRANSFERABILITY_THRESHOLD = 0.60

# Gap analysis bucketing thresholds
MATCHED_THRESHOLD = 0.95
TRANSFERABLE_THRESHOLD = 0.55

# How much the preferred (nice-to-have) skills count inside hard_skills, by
# role level. A student posting lists its stack as a wish-list and expects
# most applicants to miss parts of it; a senior posting expects most of it.
# Required-skill coverage always carries the remaining share.
PREFERRED_SKILL_WEIGHT_BY_LEVEL: Dict[str, float] = {
    "student": 0.15,
    "junior": 0.20,
    "mid": 0.25,
    "senior": 0.30,
}

# Nice-to-haves can lift hard_skills at most this far above required-skill
# coverage, so a candidate missing the one must-have never outscores a
# candidate who has it.
REQUIRED_COVERAGE_HEADROOM = 0.15

# Eligibility item scores by status. 'unclear' (the CV has no information) is
# not scored at all: discounting it lowered the score for something the CV
# simply doesn't say, so it is reported as an open question instead.
ELIGIBILITY_STATUS_SCORE: Dict[str, float] = {"met": 1.0, "unmet": 0.0}

# Raw cosine between two long-text embeddings rarely leaves [0.25, 0.80] for
# text-embedding-3-small; stretch that band onto [0, 1] so domain fit moves
# the aggregate instead of sitting near 0.5 for everyone.
DOMAIN_COSINE_FLOOR = 0.25
DOMAIN_COSINE_CEIL = 0.80


# ============================================
# Internal Helper Functions
# ============================================

def _band_for_score(score: float) -> str:
    for upper, label in COMPATIBILITY_BANDS:
        if score < upper:
            return label
    return "excellent"


def _flatten_cv_skill_mentions(cv_data: Dict[str, Any]) -> List[str]:
    """
    Build a flat list of strings that surface skill mentions on the CV:
    declared skills + experience text + project descriptions. Used to detect
    direct-skill mentions even when they aren't in the dedicated skills list.
    """
    mentions: List[str] = []

    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(_text(v) for v in value if v)
        if isinstance(value, dict):
            return " ".join(_text(v) for v in value.values() if v)
        return str(value)

    def _entries(value: Any, fields: List[str]) -> None:
        if isinstance(value, str):
            mentions.append(value)
            return
        if not isinstance(value, list):
            return
        for entry in value:
            if isinstance(entry, str):
                mentions.append(entry)
            elif isinstance(entry, dict):
                joined = " ".join(_text(entry.get(f)) for f in fields)
                mentions.append(joined)

    summary = cv_data.get("summary")
    if isinstance(summary, str) and summary.strip():
        mentions.append(summary)

    skills = cv_data.get("skills") or []
    if isinstance(skills, list):
        mentions.extend(str(s) for s in skills if s)
    elif isinstance(skills, str):
        mentions.append(skills)

    _entries(cv_data.get("experience") or cv_data.get("experiences") or [],
             ["position", "title", "company", "responsibilities"])
    # Project technologies and outcomes are where students keep most of their
    # stack; the old flattener read only name + description and missed them.
    _entries(cv_data.get("projects") or [], ["name", "description", "technologies", "outcomes"])
    _entries(cv_data.get("education") or [], ["degree", "institution", "details"])
    _entries(cv_data.get("leadership_activities") or [], ["role", "organization", "description", "highlights"])
    _entries(cv_data.get("languages") or [], ["language", "fluency"])

    return [m for m in mentions if m and m.strip()]


def _candidate_skills_normalized(cv_data: Dict[str, Any]) -> List[str]:
    """Return the candidate's declared skill list, raw (not normalized)."""
    skills = cv_data.get("skills") or []
    if isinstance(skills, list):
        return [str(s) for s in skills if s and str(s).strip()]
    if isinstance(skills, str):
        return [skills]
    return []


def _detect_direct_match(required_skill: str, cv_data: Dict[str, Any]) -> Optional[str]:
    """
    Look for the required skill in the candidate's declared skills first
    (preferred match), then in free-text mentions across experience/projects.
    Returns the matched candidate skill string if found, else None.
    """
    pattern = _skill_mention_pattern(required_skill)
    if pattern is None:
        return None

    declared = _candidate_skills_normalized(cv_data)
    for cand in declared:
        if skill_graph_service.is_direct_match(required_skill, cand):
            return cand
    # A declared entry such as 'Java (advanced)' or 'Python, Django' still
    # names the skill; alias equality alone would miss it.
    for cand in declared:
        if pattern.search(cand):
            return cand

    for blob in _flatten_cv_skill_mentions(cv_data):
        if pattern.search(blob):
            return required_skill
    return None


@lru_cache(maxsize=2048)
def _skill_mention_pattern(skill: str) -> Optional["re.Pattern[str]"]:
    """
    Regex that finds any surface form of `skill` in free text.

    Uses lookarounds instead of \\b so names ending in a symbol ('C++', 'C#')
    or containing a dot ('Node.js') still match at their edges, while 'c'
    never matches inside 'c++' and 'java' never matches inside 'javascript'.
    """
    forms = skill_graph_service.surface_forms(skill)
    if not forms:
        return None
    alternation = "|".join(re.escape(f) for f in forms)
    return re.compile(rf"(?<![\w+#.])(?:{alternation})(?![\w+#])", re.IGNORECASE)


def _detect_family_match(
    required_skill: str, cv_data: Dict[str, Any]
) -> Optional[Tuple[str, float, str]]:
    """Returns (matched_candidate_skill, transferability, family_name) or None."""
    declared = _candidate_skills_normalized(cv_data)
    return skill_graph_service.best_family_match(required_skill, declared)


def _embedding_transferability(
    required_skill: str, candidate_skills: List[str]
) -> Optional[Tuple[str, float]]:
    """
    Compute embedding cosine between required skill and each declared candidate
    skill. Returns the best (candidate_skill, mapped_score) if best cosine
    crosses EMBEDDING_TRANSFERABILITY_THRESHOLD, else None.

    The raw cosine is mapped onto [0.55, 0.80] so embedding-only matches never
    edge into the matched_skills bucket reserved for direct/family matches.
    """
    if not candidate_skills:
        return None
    # When the curated graph places both skills in *different* families, that
    # is a deliberate 'not the same thing' verdict (Java vs JavaScript, C vs
    # C#). Raw cosine on such short names is unreliable, so leave those pairs
    # to the LLM judge instead of letting a lexical neighbour score them.
    required_family = skill_graph_service.find_family(required_skill)
    if required_family:
        candidate_skills = [
            c for c in candidate_skills
            if skill_graph_service.find_family(c) in (None, required_family)
        ]
        if not candidate_skills:
            return None
    try:
        req_emb = get_skill_embedding(required_skill)
        if not req_emb:
            return None
        best_cand: Optional[str] = None
        best_cos = 0.0
        for cand in candidate_skills:
            cand_emb = get_skill_embedding(cand)
            if not cand_emb:
                continue
            cos = cosine_similarity(req_emb, cand_emb)
            if cos > best_cos:
                best_cos = cos
                best_cand = cand
        if best_cand is None or best_cos < EMBEDDING_TRANSFERABILITY_THRESHOLD:
            return None
        span = 1.0 - EMBEDDING_TRANSFERABILITY_THRESHOLD
        if span <= 0:
            mapped = 0.55
        else:
            mapped = 0.55 + ((best_cos - EMBEDDING_TRANSFERABILITY_THRESHOLD) / span) * 0.25
        mapped = max(0.55, min(0.80, mapped))
        return (best_cand, mapped)
    except Exception:
        return None


# ============================================
# LLM Transferability Assessment
# ============================================

TransferVerdict = Literal["same_family", "adjacent", "distant", "none"]

# The judge names a category; the score comes from this table, not from the
# model. A free 0-1 number put the judge's "adjacent" band (0.55-0.75) right
# on TRANSFERABLE_THRESHOLD, so any adjacent skill - Spring Boot via Django -
# counted as covered. Only same_family clears the threshold; adjacent is a
# learnable gap with partial credit.
VERDICT_SCORES: Dict[str, Tuple[str, float]] = {
    "same_family": ("transferable", 0.80),
    "adjacent": ("learnable", 0.45),
    "distant": ("missing", 0.25),
    "none": ("missing", 0.0),
}


class _LLMTransferabilityItem(BaseModel):
    """Schema for a single LLM-judged transferability assessment."""
    required_skill: str = Field(description="The required skill being assessed")
    best_candidate_skill: Optional[str] = Field(
        default=None,
        description="Best candidate skill that relates to the required skill, if any"
    )
    verdict: TransferVerdict = Field(
        description=(
            "same_family: interchangeable tools of one kind (React/Vue, Postgres/MySQL); "
            "adjacent: related but a different paradigm or ecosystem, real learning needed "
            "(Django -> Spring Boot, Python -> Go); distant: same broad field only; "
            "none: unrelated or nothing on the CV"
        )
    )
    rationale: str = Field(description="Short explanation of the judgment")


class _LLMTransferabilityBatch(BaseModel):
    """Schema for batched LLM transferability assessments."""
    assessments: List[_LLMTransferabilityItem] = Field(default_factory=list)


def _assess_transferability_llm(
    unresolved_required: List[str],
    candidate_skills: List[str],
    cv_summary_text: str,
) -> Dict[str, _LLMTransferabilityItem]:
    """
    Single batched call to GPT-4o-mini that judges transferability for every
    unresolved required skill at once. Returns a dict keyed by required skill.

    This is the only LLM call introduced by v2 scoring per run, in line with
    the cost target of ~1-2 extra LLM calls per compatibility check.
    """
    if not unresolved_required:
        return {}

    llm = chat_openai_mini(temperature=0)
    structured_llm = llm.with_structured_output(_LLMTransferabilityBatch)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior staff engineer and technical recruiter.

For each required skill, decide how the candidate's declared skills and
experience relate to it. Think in terms of underlying paradigms and learning
curve, not surface keywords, and pick exactly one verdict:

- same_family: the candidate holds an interchangeable tool of the same kind,
  productive within days (React <-> Vue, PyTorch <-> TensorFlow,
  Postgres <-> MySQL).
- adjacent: related, but a different paradigm, language, or ecosystem; real
  learning is needed (Django -> Spring Boot, Python -> Go, REST -> GraphQL,
  Java -> Kotlin).
- distant: only the same broad field (frontend -> embedded, data analysis ->
  backend APIs).
- none: unrelated, or the CV shows nothing relevant.

The candidate does not hold the required skill itself; skills they list
verbatim were matched before this step. When in doubt between two verdicts,
choose the lower one: overstating coverage misleads the candidate more than
understating it.

For each item return:
- required_skill: echo the input exactly
- best_candidate_skill: the single most relevant candidate skill (or null)
- verdict: one of same_family, adjacent, distant, none
- rationale: 1-2 sentences naming what the candidate has and what they'd need to learn
"""),
        ("user", """Candidate declared skills:
{candidate_skills}

Candidate experience summary:
{cv_summary}

Required skills to assess:
{required}

Return a transferability assessment for each required skill.""")
    ])

    chain = prompt | structured_llm
    try:
        result: _LLMTransferabilityBatch = chain.invoke({
            "candidate_skills": ", ".join(candidate_skills) if candidate_skills else "(none)",
            "cv_summary": cv_summary_text[:4000],
            "required": "\n".join(f"- {s}" for s in unresolved_required),
        })
        # Key on the normalized echo: the model may return 'spring boot' for
        # 'Spring Boot', and an exact-string lookup would silently drop it.
        return {_norm_key(item.required_skill): item for item in result.assessments}
    except Exception:
        return {}


# ============================================
# Skill Matching Cascade
# ============================================

def _norm_key(text: Any) -> str:
    """Lookup key for skill names: casefolded, whitespace-collapsed."""
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _technical_skill_lists(job_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Required / preferred technical skill lists with eligibility items removed.

    The extraction prompt asks for languages and education to go into their
    own fields, but models still occasionally drop 'German' or "Bachelor's
    degree" into required_skills. Those would then be 'missing' technical
    skills nobody can bridge; route them to the eligibility check instead.
    """
    def _clean(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        return [str(s).strip() for s in raw if s and str(s).strip()]

    required = [s for s in _clean(job_data.get("required_skills") or job_data.get("skills"))
                if _eligibility_kind(s) is None]
    preferred = [s for s in _clean(job_data.get("preferred_skills"))
                 if _eligibility_kind(s) is None]
    return required, preferred


def _collapse_required_skill_groups(
    matches: List[SkillMatch],
    groups: List[List[str]],
) -> List[SkillMatch]:
    """
    Collapse OR-alternative requirement groups into a single SkillMatch per
    group, so a candidate who satisfies ANY ONE alternative isn't penalized
    for lacking the others.

    Without this, a posting phrased as "Java and WildFly/Quarkus/Spring
    Boot" - which is really two requirements, Java and *any one* JVM
    framework - gets flattened into four independent required skills, and a
    candidate with Java + Spring Boot alone shows two "genuine gaps" (WildFly,
    Quarkus) for a requirement they've fully satisfied.

    Required matches not covered by any group pass through unchanged (a
    posting is not required to group everything); preferred matches are
    never touched, since grouping only applies to hard requirements.
    """
    if not groups:
        return matches

    by_name = {m.required_skill: m for m in matches if m.is_required}
    grouped_names: set[str] = set()
    collapsed: List[SkillMatch] = []

    for group in groups:
        group_matches = [by_name[name] for name in group if name in by_name]
        if not group_matches:
            continue
        grouped_names.update(group)
        if len(group_matches) == 1:
            collapsed.append(group_matches[0])
            continue

        best = max(group_matches, key=lambda m: m.transferability)
        label = " / ".join(group)
        collapsed.append(best.model_copy(update={
            "required_skill": label,
            "rationale": (
                f"Posting accepts any of {label} for this requirement; "
                f"strongest match is '{best.matched_with}' for '{best.required_skill}'. "
                if best.matched_with else
                f"Posting accepts any of {label} for this requirement; none matched. "
            ) + best.rationale,
        }))

    # Every match not swallowed into a group above - required-but-ungrouped
    # skills, and all preferred skills - passes through untouched.
    passthrough = [m for m in matches if not (m.is_required and m.required_skill in grouped_names)]
    return passthrough + collapsed


def _match_from_verdict(unresolved: SkillMatch, judged: "_LLMTransferabilityItem") -> SkillMatch:
    """
    Turn the judge's verdict into a SkillMatch.

    The curated skill graph outranks the judge: when it places the required
    skill and the candidate's skill in different families (Spring Boot in
    JVM frameworks, Django in Python frameworks), they are not the same
    thing, so "same_family" is capped at "adjacent" - learnable, not covered.
    """
    verdict = judged.verdict
    candidate = judged.best_candidate_skill
    if verdict == "same_family" and candidate:
        required_family = skill_graph_service.find_family(unresolved.required_skill)
        candidate_family = skill_graph_service.find_family(candidate)
        if required_family and candidate_family and required_family != candidate_family:
            verdict = "adjacent"
    if not candidate and verdict in ("same_family", "adjacent"):
        verdict = "distant"  # nothing on the CV to build on
    kind, score = VERDICT_SCORES[verdict]
    return SkillMatch(
        required_skill=unresolved.required_skill,
        matched_with=candidate,
        kind=kind,
        transferability=score,
        rationale=judged.rationale,
        bridge_bullet=None,
        is_required=unresolved.is_required,
    )


def _match_skill_pairs_internal(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[SkillMatch]:
    """
    Resolve every required and preferred job skill against the candidate's CV
    using the cascade: direct -> family -> embedding -> LLM.
    """
    required_list, preferred_list = _technical_skill_lists(job_data)

    declared_candidate = _candidate_skills_normalized(cv_data)
    cv_summary_parts = _flatten_cv_skill_mentions(cv_data)
    cv_summary = " ".join(cv_summary_parts)

    unresolved: List[Tuple[str, bool]] = []  # (skill, is_required)
    matches: List[SkillMatch] = []

    def resolve(skill: str, is_required: bool) -> SkillMatch:
        # 1. Direct match
        direct = _detect_direct_match(skill, cv_data)
        if direct:
            return SkillMatch(
                required_skill=skill,
                matched_with=direct,
                kind="direct",
                transferability=1.0,
                rationale=f"Candidate explicitly lists or mentions '{direct}'.",
                bridge_bullet=None,
                is_required=is_required,
            )

        # 2. Family match via skill graph
        fam = _detect_family_match(skill, cv_data)
        if fam:
            cand, score, family_name = fam
            paradigm = skill_graph_service.family_paradigm(family_name) or family_name
            return SkillMatch(
                required_skill=skill,
                matched_with=cand,
                kind="family",
                transferability=float(score),
                rationale=(
                    f"'{cand}' is in the same family as '{skill}' "
                    f"({paradigm}); skills in this family share core paradigms."
                ),
                bridge_bullet=None,
                is_required=is_required,
            )

        # 3. Embedding fallback
        emb = _embedding_transferability(skill, declared_candidate)
        if emb is not None:
            cand, mapped = emb
            return SkillMatch(
                required_skill=skill,
                matched_with=cand,
                kind="transferable",
                transferability=float(mapped),
                rationale=(
                    f"'{cand}' is semantically related to '{skill}' based on "
                    "embedding similarity; transferability needs validation."
                ),
                bridge_bullet=None,
                is_required=is_required,
            )

        return SkillMatch(
            required_skill=skill,
            matched_with=None,
            kind="missing",
            transferability=0.0,
            rationale="No direct, family, or semantic match found in candidate skills.",
            bridge_bullet=None,
            is_required=is_required,
        )

    for skill in required_list:
        m = resolve(skill, True)
        matches.append(m)
        if m.kind == "missing":
            unresolved.append((skill, True))

    for skill in preferred_list:
        m = resolve(skill, False)
        matches.append(m)
        if m.kind == "missing":
            unresolved.append((skill, False))

    # 4. Single batched LLM call for unresolved skills
    if unresolved:
        unresolved_names = [s for s, _ in unresolved]
        llm_results = _assess_transferability_llm(
            unresolved_names, declared_candidate, cv_summary
        )
        if llm_results:
            for idx, m in enumerate(matches):
                if m.kind != "missing":
                    continue
                judged = llm_results.get(_norm_key(m.required_skill))
                if judged is None:
                    continue
                matches[idx] = _match_from_verdict(m, judged)

    # 5. Collapse OR-alternative requirement groups (see
    # `required_skill_groups` on JobRequirements) AFTER every skill has been
    # through the full cascade, so the group's representative is whichever
    # alternative actually scored best - not an arbitrary pick made before
    # the LLM had a chance to assess every option.
    groups_raw = job_data.get("required_skill_groups") or []
    normalized_groups = [
        [str(s).strip() for s in g if s and str(s).strip()]
        for g in groups_raw if isinstance(g, list) and g
    ]
    if normalized_groups:
        matches = _collapse_required_skill_groups(matches, normalized_groups)

    return matches


@tool
def match_skill_pairs(cv_json: str, job_json: str) -> str:
    """
    Match every required and preferred job skill against the candidate's CV.

    Uses a cascade (cheapest first):
    1. Direct/alias match against declared skills or free-text mentions.
    2. Skill-graph family match (e.g. React<->Vue, PyTorch<->TensorFlow).
    3. Embedding cosine fallback for semantic neighbors.
    4. Single batched LLM call for whatever remains, with bridge bullet drafts.

    Args:
        cv_json: ResumeInfo JSON string from cv_agent.
        job_json: JobRequirements JSON string from job_agent.

    Returns:
        JSON list of SkillMatch objects with kind, transferability, rationale,
        and (for transferable skills) a draft bridge bullet.
    """
    try:
        cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
        job_data = json.loads(job_json) if isinstance(job_json, str) else job_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "matches": []})

    matches = _match_skill_pairs_internal(cv_data, job_data)
    return json.dumps([m.model_dump() for m in matches], indent=2)


@tool
def assess_transferability_llm(unresolved_pairs_json: str) -> str:
    """
    Batched LLM transferability assessment for skills that didn't resolve via
    direct match, skill-graph family lookup, or embedding similarity.

    This tool is exposed so the agent can call it standalone when needed. In
    the normal v2 scoring flow it is invoked internally by match_skill_pairs.

    Args:
        unresolved_pairs_json: JSON object with keys
            - required: list[str] of required skills to assess
            - candidate_skills: list[str] of declared candidate skills
            - cv_summary: optional string of free-text experience summary

    Returns:
        JSON list of assessments, each with required_skill, best_candidate_skill,
        verdict (same_family / adjacent / distant / none; scored via
        VERDICT_SCORES), and rationale.
    """
    try:
        payload = json.loads(unresolved_pairs_json) if isinstance(unresolved_pairs_json, str) else unresolved_pairs_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "assessments": []})

    required = payload.get("required", []) or []
    candidate_skills = payload.get("candidate_skills", []) or []
    cv_summary = payload.get("cv_summary", "") or ""

    results = _assess_transferability_llm(
        [str(s) for s in required if s],
        [str(s) for s in candidate_skills if s],
        str(cv_summary),
    )
    return json.dumps([item.model_dump() for item in results.values()], indent=2)


# ============================================
# Eligibility — languages, degree, enrollment
# ============================================
#
# These are the requirements that decide whether an application is viable at
# all (a Werkstudent posting needs an enrolled student who speaks German),
# and they are not technical skills: nothing in the skill graph, embeddings,
# or the rewrite step can bridge them. They get their own check with three
# outcomes - met / unclear / unmet - where 'unclear' means the CV simply has
# no information (no languages section), which is a prompt to add it, not a
# verdict against the candidate.

_LANGUAGE_CANONICAL: Dict[str, str] = {
    "german": "german", "deutsch": "german",
    "english": "english", "englisch": "english",
    "french": "french", "französisch": "french", "francais": "french", "français": "french",
    "spanish": "spanish", "spanisch": "spanish", "español": "spanish",
    "italian": "italian", "italienisch": "italian",
    "dutch": "dutch", "niederländisch": "dutch", "nederlands": "dutch",
    "portuguese": "portuguese", "portugiesisch": "portuguese",
    "russian": "russian", "russisch": "russian",
    "polish": "polish", "polnisch": "polish",
    "turkish": "turkish", "türkisch": "turkish",
    "chinese": "chinese", "mandarin": "chinese", "chinesisch": "chinese",
    "japanese": "japanese", "japanisch": "japanese",
    "korean": "korean", "koreanisch": "korean",
    "arabic": "arabic", "arabisch": "arabic",
    "hindi": "hindi",
    "vietnamese": "vietnamese", "vietnamesisch": "vietnamese",
    "swedish": "swedish", "schwedisch": "swedish",
    "danish": "danish", "dänisch": "danish",
    "norwegian": "norwegian", "norwegisch": "norwegian",
    "finnish": "finnish", "finnisch": "finnish",
    "czech": "czech", "tschechisch": "czech",
    "greek": "greek", "griechisch": "greek",
    "ukrainian": "ukrainian", "ukrainisch": "ukrainian",
    "romanian": "romanian", "rumänisch": "romanian",
    "hungarian": "hungarian", "ungarisch": "hungarian",
}

# Proficiency ordinal: CEFR levels plus the words postings and CVs use.
_LANGUAGE_LEVELS: List[Tuple["re.Pattern[str]", int]] = [
    (re.compile(r"\b(?:native|mother tongue|muttersprache|muttersprachlich|muttersprachler(?:in)?)\b", re.I), 7),
    (re.compile(r"\bc2\b", re.I), 6),
    (re.compile(r"\b(?:c1|fluent|fließend|fliessend|verhandlungssicher|business fluent|full professional|proficient|advanced|sehr gut)\b", re.I), 5),
    (re.compile(r"\b(?:b2|upper intermediate|professional working|gut|good)\b", re.I), 4),
    (re.compile(r"\b(?:b1|intermediate|conversational|limited working)\b", re.I), 3),
    (re.compile(r"\b(?:a2|basic|elementary|grundkenntnisse)\b", re.I), 2),
    (re.compile(r"\b(?:a1|beginner|anfänger)\b", re.I), 1),
]

_LEVEL_LABEL = {7: "native", 6: "C2", 5: "C1/fluent", 4: "B2", 3: "B1", 2: "A2", 1: "A1"}

_EDUCATION_HINT_RE = re.compile(
    r"\b(?:degree|bachelor(?:'s)?|master(?:'s)?|b\.?\s?sc\.?|m\.?\s?sc\.?|b\.?\s?eng|m\.?\s?eng|phd|ph\.d|doctorate|"
    r"diploma|diplom|abschluss|studium|student(?:in)?|enrolled|eingeschrieben|immatrikuliert|"
    r"university|hochschule|graduate|studies|studying)\b",
    re.I,
)
_ENROLLMENT_RE = re.compile(
    r"\b(?:enrolled|enrolment|enrollment|student(?:in)?|studying|currently studying|eingeschrieben|"
    r"immatrikuliert|werkstudent(?:in)?|studium|im studium)\b",
    re.I,
)
_ONGOING_DATES_RE = re.compile(
    r"\b(?:expected|present|current|now|today|ongoing|voraussichtlich|heute|laufend|seit)\b", re.I
)
_DEGREE_LEVEL_PATTERNS: List[Tuple["re.Pattern[str]", int]] = [
    (re.compile(r"\b(?:phd|ph\.d|doctorate|doktor|promotion)\b", re.I), 3),
    (re.compile(r"\b(?:master(?:'s)?|m\.?\s?sc\.?|m\.?\s?eng|m\.?\s?a\.?|msc|meng|mba|diplom)\b", re.I), 2),
    (re.compile(r"\b(?:bachelor(?:'s)?|b\.?\s?sc\.?|b\.?\s?eng|b\.?\s?a\.?|bsc|beng|undergraduate)\b", re.I), 1),
]

# Field-of-study families: a requirement naming any term of a family is met
# by a CV degree naming any term of the same family.
_FIELD_FAMILIES: Dict[str, List[str]] = {
    "computer science": [
        "computer science", "informatik", "informatics", "software engineering", "software",
        "computer engineering", "information systems", "wirtschaftsinformatik", "information technology",
        "it", "computing", "data science", "artificial intelligence", "machine learning", "cs",
        "technische informatik", "angewandte informatik", "medieninformatik", "cybersecurity",
    ],
    "engineering": [
        "engineering", "ingenieur", "elektrotechnik", "electrical", "mechanical", "maschinenbau",
        "mechatronics", "mechatronik", "automation", "robotics",
    ],
    "mathematics": ["mathematics", "mathematik", "math", "statistics", "statistik", "physics", "physik"],
    "business": [
        "business", "economics", "bwl", "betriebswirtschaft", "volkswirtschaft", "management",
        "finance", "accounting", "marketing", "wirtschaft",
    ],
    "design": ["design", "hci", "human-computer interaction", "media", "ux", "user experience"],
    "natural sciences": ["biology", "biologie", "chemistry", "chemie", "life sciences", "bioinformatics"],
}
_RELATED_FIELD_RE = re.compile(
    r"\b(?:related|similar|comparable|equivalent|vergleichbar|ähnlich|verwandt|or similar|oder ähnlich)\b", re.I
)


def _eligibility_kind(text: str) -> Optional[str]:
    """
    'language' if the text names a spoken language (optionally with a level),
    'education' if it reads like a degree/enrollment requirement, else None.
    Used both to classify job fields and to catch misfiled required_skills.
    """
    if not text:
        return None
    low = text.casefold()
    if _parse_language(low)[0] is not None:
        return "language"
    if _EDUCATION_HINT_RE.search(low):
        return "education"
    return None


def _parse_language(text: str) -> Tuple[Optional[str], Optional[int]]:
    """(canonical language, level ordinal or None) for a requirement or CV entry."""
    if not text:
        return None, None
    low = str(text).casefold()
    # Strip parentheses, colons etc so 'German (C1)' and 'Deutsch: fließend' parse.
    words = re.findall(r"[a-zäöüß]+", low)
    language = None
    for w in words:
        canon = _LANGUAGE_CANONICAL.get(w)
        if canon:
            language = canon
            break
    if language is None:
        return None, None
    level = None
    for pattern, value in _LANGUAGE_LEVELS:
        if pattern.search(low):
            level = value
            break
    return language, level


def _cv_languages(cv_data: Dict[str, Any]) -> List[Tuple[str, Optional[int], str]]:
    """(canonical, level, original text) for every language the CV states."""
    found: List[Tuple[str, Optional[int], str]] = []
    raw = cv_data.get("languages") or []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                text = " ".join(str(item.get(k) or "") for k in ("language", "fluency", "level", "proficiency"))
            else:
                text = str(item)
            canon, level = _parse_language(text)
            if canon:
                found.append((canon, level, text.strip()))
    # Older extractions crammed 'German (native)' into skills; honour those too.
    skills = cv_data.get("skills") or []
    if isinstance(skills, list):
        for s in skills:
            canon, level = _parse_language(str(s))
            if canon and _eligibility_kind(str(s)) == "language":
                found.append((canon, level, str(s).strip()))
    return found


def _check_language(requirement: str, cv_langs: List[Tuple[str, Optional[int], str]]) -> EligibilityCheck:
    want, want_level = _parse_language(requirement)
    if want is None:
        return EligibilityCheck(requirement=requirement, kind="language", status="unclear",
                                rationale="Could not identify which language is required.")
    if not cv_langs:
        return EligibilityCheck(
            requirement=requirement, kind="language", status="unclear",
            rationale="CV lists no languages. Add a Languages section so this requirement can be confirmed.",
        )
    have = [(lvl, text) for canon, lvl, text in cv_langs if canon == want]
    if not have:
        return EligibilityCheck(
            requirement=requirement, kind="language", status="unmet",
            rationale=f"CV lists languages but not {want.title()}.",
        )
    best_level, best_text = max(have, key=lambda h: (h[0] if h[0] is not None else -1))
    if want_level is None or best_level is None:
        note = "" if want_level is None else f" Required level {_LEVEL_LABEL.get(want_level, want_level)}; CV states no level, so it is assumed met."
        return EligibilityCheck(requirement=requirement, kind="language", status="met",
                                matched_with=best_text, rationale=f"CV lists {best_text}.{note}")
    if best_level >= want_level:
        return EligibilityCheck(requirement=requirement, kind="language", status="met",
                                matched_with=best_text, rationale=f"CV lists {best_text}, at or above the required level.")
    return EligibilityCheck(
        requirement=requirement, kind="language", status="unmet", matched_with=best_text,
        rationale=(f"CV states {best_text} ({_LEVEL_LABEL.get(best_level, best_level)}), below the required "
                   f"{_LEVEL_LABEL.get(want_level, want_level)}."),
    )


def _cv_education_entries(cv_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(searchable text, display label) for each education entry (dict or str)."""
    out: List[Tuple[str, str]] = []
    raw = cv_data.get("education") or []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if isinstance(entry, dict):
            degree = str(entry.get("degree") or "")
            institution = str(entry.get("institution") or "")
            dates = str(entry.get("dates") or entry.get("duration") or "")
            details = " ".join(str(d) for d in (entry.get("details") or []) if d)
            text = " ".join(p for p in (degree, institution, dates, details) if p)
            label = degree or institution or text[:60]
        else:
            text = str(entry)
            label = text[:60]
        if text.strip():
            out.append((text, label))
    return out


def _fields_in(text: str) -> set:
    low = text.casefold()
    hits = set()
    for family, terms in _FIELD_FAMILIES.items():
        for term in terms:
            if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", low):
                hits.add(family)
                break
    return hits


def _degree_level_in(text: str) -> Optional[int]:
    levels = [lvl for pattern, lvl in _DEGREE_LEVEL_PATTERNS if pattern.search(text)]
    return max(levels) if levels else None


def _entry_is_ongoing(text: str) -> bool:
    if _ONGOING_DATES_RE.search(text):
        return True
    years = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    return bool(years) and max(years) >= datetime.now().year


def _check_education(requirement: str, entries: List[Tuple[str, str]]) -> EligibilityCheck:
    req_low = requirement.casefold()
    wants_enrollment = bool(_ENROLLMENT_RE.search(req_low))
    want_fields = _fields_in(requirement)
    want_level = _degree_level_in(requirement)
    related_ok = bool(_RELATED_FIELD_RE.search(req_low))

    if not wants_enrollment and not want_fields and want_level is None:
        return EligibilityCheck(requirement=requirement, kind="education", status="unclear",
                                rationale="Requirement is too vague to check automatically.")
    if not entries:
        return EligibilityCheck(requirement=requirement, kind="education", status="unclear",
                                rationale="CV lists no education entries.")

    best: Optional[Tuple[int, str, List[str]]] = None  # (score, label, problems)
    for text, label in entries:
        problems: List[str] = []
        if want_fields:
            have_fields = _fields_in(text)
            if not (have_fields & want_fields) and not (related_ok and have_fields):
                problems.append(f"field of study does not match ({', '.join(sorted(want_fields))})")
        if want_level is not None:
            have_level = _degree_level_in(text)
            # An enrolled Bachelor student satisfies "student in CS"; a
            # completed-degree requirement needs the level itself.
            if not wants_enrollment and (have_level is None or have_level < want_level):
                problems.append("degree level below requirement")
        if wants_enrollment and not _entry_is_ongoing(text):
            problems.append("no ongoing enrollment visible (no 'expected'/'present' date)")
        score = 3 - len(problems)
        if best is None or score > best[0]:
            best = (score, label, problems)

    assert best is not None
    score, label, problems = best
    if not problems:
        return EligibilityCheck(requirement=requirement, kind="education", status="met",
                                matched_with=label, rationale=f"Satisfied by '{label}'.")
    # Only a field mismatch or level shortfall is a real miss; a missing
    # enrollment date is more often a CV formatting gap than a fact.
    status = "unmet" if any(not p.startswith("no ongoing") for p in problems) else "unclear"
    return EligibilityCheck(requirement=requirement, kind="education", status=status,
                            matched_with=label, rationale=f"Closest entry '{label}': {'; '.join(problems)}.")


def _check_eligibility(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[EligibilityCheck]:
    """Every language / education requirement on the posting, checked against the CV."""
    def _clean(raw: Any) -> List[str]:
        return [str(s).strip() for s in raw if s and str(s).strip()] if isinstance(raw, list) else []

    languages = _clean(job_data.get("required_languages"))
    education = _clean(job_data.get("education_requirements"))
    # Misfiled items from the skill lists (see _technical_skill_lists).
    known_languages = {_parse_language(l)[0] for l in languages}
    for s in _clean(job_data.get("required_skills") or job_data.get("skills")):
        kind = _eligibility_kind(s)
        if kind == "language":
            canon = _parse_language(s)[0]
            if canon not in known_languages:
                languages.append(s)
                known_languages.add(canon)
        elif kind == "education" and _norm_key(s) not in {_norm_key(e) for e in education}:
            education.append(s)

    checks: List[EligibilityCheck] = []
    if languages:
        cv_langs = _cv_languages(cv_data)
        checks.extend(_check_language(req, cv_langs) for req in languages)
    if education:
        entries = _cv_education_entries(cv_data)
        checks.extend(_check_education(req, entries) for req in education)
    return checks


def _score_eligibility(checks: List[EligibilityCheck]) -> Tuple[Optional[float], str]:
    """
    Score over the requirements the CV lets us decide. Returns a None score
    when nothing is decidable (no requirements, or all unclear), and the
    caller then leaves the dimension out of the weighted sum.
    """
    unclear = [c.requirement for c in checks if c.status == "unclear"]
    decided = [c for c in checks if c.status != "unclear"]
    if not decided:
        if not checks:
            return None, "Posting states no language or education requirements."
        return None, f"Not scored until confirmed (the CV says nothing about): {', '.join(unclear)}."
    score = sum(ELIGIBILITY_STATUS_SCORE[c.status] for c in decided) / len(decided)
    met = [c.requirement for c in decided if c.status == "met"]
    unmet = [c.requirement for c in decided if c.status == "unmet"]
    parts = [f"{len(met)} of {len(decided)} checked requirements met."]
    if unmet:
        parts.append(f"Unmet: {', '.join(unmet)}.")
    if unclear:
        parts.append(f"Not scored until confirmed: {', '.join(unclear)}.")
    return score, " ".join(parts)


# ============================================
# Dimension Scorers
# ============================================

SkillBucket = Literal["matched", "transferable", "learnable", "missing"]


def _bucket(match: SkillMatch) -> SkillBucket:
    """The GapAnalysis bucket of one match. Learnable is decided by kind, not
    score, so a learnable skill never drifts into 'transferable' (covered)."""
    if match.kind == "learnable":
        return "learnable"
    if match.transferability >= MATCHED_THRESHOLD:
        return "matched"
    if match.transferability >= TRANSFERABLE_THRESHOLD:
        return "transferable"
    return "missing"


def _bucket_counts(matches: List[SkillMatch]) -> Tuple[int, int, int, int]:
    """(matched, transferable, learnable, missing), the same split as GapAnalysis."""
    buckets = [_bucket(m) for m in matches]
    return (buckets.count("matched"), buckets.count("transferable"),
            buckets.count("learnable"), buckets.count("missing"))


def _bucket_summary(matches: List[SkillMatch]) -> str:
    matched, transferable, learnable, missing = _bucket_counts(matches)
    parts = [f"{matched} matched", f"{transferable} transferable"]
    if learnable:
        parts.append(f"{learnable} learnable")
    parts.append(f"{missing} missing")
    return ", ".join(parts)


def _score_hard_skills(matches: List[SkillMatch], job_level: str = "") -> Tuple[float, str]:
    """
    Required-skill coverage first, nice-to-haves as a level-scaled bonus.

    - required coverage = mean transferability over required skills (after
      OR-group collapse); this is the backbone of the score.
    - preferred coverage = mean over preferred skills, blended in with a
      weight that depends on the role level (student postings barely expect
      their wish-list; senior postings expect most of it).
    - the blend can lift the score at most REQUIRED_COVERAGE_HEADROOM above
      required coverage, so missing the one must-have is never rescued by
      collecting nice-to-haves.
    """
    if not matches:
        return 0.0, "No required or preferred skills declared on the job."

    required = [m for m in matches if m.is_required]
    preferred = [m for m in matches if not m.is_required]
    req_cov = sum(m.transferability for m in required) / len(required) if required else None
    pref_cov = sum(m.transferability for m in preferred) / len(preferred) if preferred else None

    bucket = _level_bucket(job_level)
    pw = PREFERRED_SKILL_WEIGHT_BY_LEVEL.get(bucket, PREFERRED_SKILL_WEIGHT_BY_LEVEL["mid"])

    if req_cov is None:
        score = pref_cov or 0.0
    elif pref_cov is None:
        score = req_cov
    else:
        score = req_cov * (1.0 - pw) + pref_cov * pw
        score = min(score, req_cov + REQUIRED_COVERAGE_HEADROOM)
    score = max(0.0, min(1.0, score))

    parts: List[str] = []
    if required:
        parts.append(f"Required: {_bucket_summary(required)} of {len(required)}.")
    if preferred:
        parts.append(
            f"Preferred: {_bucket_summary(preferred)} of {len(preferred)} "
            f"(weighted {int(round(pw * 100))}% for a {bucket}-level role)."
        )
    return score, " ".join(parts)


# "3+ years of professional experience", "5 Jahre Berufserfahrung"; deliberately
# NOT a bare "N years", which also matches "10 years old codebase".
_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?|jahren?)\s+(?:of\s+)?(?:\w+\s+){0,2}?"
    r"(?:experience|exp\b|berufserfahrung|erfahrung)",
    re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(
    r"(\d{4})\s*(?:-|–|—|to|bis)\s*(present|current|now|today|heute|\d{4})", re.IGNORECASE
)


def _experience_entries(cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    experience = cv_data.get("experience") or cv_data.get("experiences") or []
    return [e for e in experience if isinstance(e, dict)] if isinstance(experience, list) else []


def _estimate_candidate_years(cv_data: Dict[str, Any]) -> float:
    """
    Rough heuristic for candidate's years of experience.

    An explicit "N years of experience" claim wins. Otherwise sum the date
    ranges of experience entries only (never education), merging overlaps so
    two concurrent part-time roles do not count double.
    """
    full_text = " ".join(_flatten_cv_skill_mentions(cv_data))
    explicit = [int(m.group(1)) for m in _YEARS_RE.finditer(full_text)]
    if explicit:
        return float(max(explicit))

    current_year = datetime.now().year
    intervals: List[Tuple[int, int]] = []
    for exp in _experience_entries(cv_data):
        text = " ".join(str(exp.get(k) or "") for k in ("duration", "dates", "start_date", "end_date"))
        for match in _DATE_RANGE_RE.finditer(text):
            try:
                start = int(match.group(1))
                end_token = match.group(2).lower()
                end = current_year if not end_token.isdigit() else int(end_token)
            except (ValueError, AttributeError):
                continue
            if start <= end <= current_year and end - start <= 50:
                intervals.append((start, end))

    total = 0.0
    covered_until: Optional[int] = None
    for start, end in sorted(intervals):
        if covered_until is not None and start < covered_until:
            start = covered_until
        if end > start:
            total += end - start
        covered_until = end if covered_until is None else max(covered_until, end)
    return total


def _score_experience(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score years_experience requirement vs estimated candidate years.

    Steep penalty if under, gentle plateau if over (recruiter intuition: a
    senior overshooting a junior role is fine; a junior undershooting is not).
    """
    required = job_data.get("years_experience")
    if required is None or not isinstance(required, (int, float)) or required <= 0:
        return 0.85, "Job did not specify a years-of-experience requirement."

    candidate_years = _estimate_candidate_years(cv_data)
    if candidate_years <= 0:
        return 0.4, f"Could not estimate candidate years; job asks for {required}+."

    ratio = candidate_years / float(required)
    if ratio >= 1.0:
        score = min(1.0, 0.85 + 0.15 * min(1.0, (ratio - 1.0) / 1.5))
    else:
        score = max(0.0, 0.85 * (ratio ** 1.6))
    return score, f"Candidate ~{candidate_years:.0f}y experience vs {required}+ required."


# Whole-word patterns, so 'Internal Tools Developer' is not an intern and
# 'Leadership' is not a lead. Student rungs are explicit: a Werkstudent
# posting or title was previously 'no seniority signal'.
_SENIORITY_PATTERNS: List[Tuple["re.Pattern[str]", int]] = [
    (re.compile(r"\b(?:werkstudent(?:in)?|working student|student|intern(?:ship)?|praktikant(?:in)?|trainee)\b", re.I), 0),
    (re.compile(r"\b(?:entry|junior|graduate)\b", re.I), 1),
    (re.compile(r"\bassociate\b", re.I), 2),
    (re.compile(r"\b(?:mid|mid-level|intermediate)\b", re.I), 3),
    (re.compile(r"\bsenior\b", re.I), 4),
    (re.compile(r"\b(?:staff|lead|manager)\b", re.I), 5),
    (re.compile(r"\b(?:principal|director|head)\b", re.I), 6),
    (re.compile(r"\b(?:vp|cto|chief)\b", re.I), 7),
]


def _seniority_to_score(level_text: str) -> Optional[int]:
    """Highest seniority rung named in the text, or None if none is."""
    if not level_text:
        return None
    hits = [score for pattern, score in _SENIORITY_PATTERNS if pattern.search(level_text)]
    return max(hits) if hits else None


def _level_bucket(job_level: str) -> str:
    """Coarse role level for weighting: student / junior / mid / senior."""
    score = _seniority_to_score(job_level or "")
    if score is None:
        return "mid"
    if score == 0:
        return "student"
    if score <= 2:
        return "junior"
    if score == 3:
        return "mid"
    return "senior"


def _score_seniority(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Compare job_level against candidate's titles. Uses a simple ordinal
    seniority scale; missing data falls back to a neutral score.
    """
    job_level = job_data.get("job_level") or job_data.get("level") or ""
    job_score = _seniority_to_score(str(job_level))

    cand_titles: List[str] = []
    experience = cv_data.get("experience") or cv_data.get("experiences") or []
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, dict):
                pos = exp.get("position") or exp.get("title")
                if pos:
                    cand_titles.append(str(pos))
            elif isinstance(exp, str):
                cand_titles.append(exp)
    cand_scores = [s for s in (_seniority_to_score(t) for t in cand_titles) if s is not None]

    if job_score is None and not cand_scores:
        return 0.7, "Neither job nor CV provided clear seniority signals; neutral score."

    if not cand_scores:
        if job_score is not None and job_score <= 1:
            return 0.8, f"Job level '{job_level}' expects little prior seniority; CV titles carry no level, which is normal here."
        return 0.5, f"Job level '{job_level}' provided but no seniority signals on CV."

    cand_max = max(cand_scores)
    if job_score is None:
        return 0.75, "Job did not specify a seniority level explicitly."

    diff = cand_max - job_score
    if diff >= 0:
        score = min(1.0, 0.85 + 0.05 * min(diff, 3))
    else:
        score = max(0.0, 0.85 + 0.20 * diff)
    return score, (
        f"Candidate top title scored {cand_max} vs job level {job_level} "
        f"(scored {job_score}); diff={diff:+d}."
    )


def _build_domain_text(data: Dict[str, Any], is_job: bool) -> str:
    if is_job:
        parts = [
            str(data.get("job_title") or ""),
            " ".join(str(r) for r in (data.get("responsibilities") or []) if r),
        ]
        return " ".join(p for p in parts if p)
    parts: List[str] = []
    experience = data.get("experience") or data.get("experiences") or []
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, dict):
                parts.append(str(exp.get("position") or ""))
                parts.append(str(exp.get("company") or ""))
            elif isinstance(exp, str):
                parts.append(exp)
    projects = data.get("projects") or []
    if isinstance(projects, list):
        for proj in projects:
            if isinstance(proj, dict):
                parts.append(str(proj.get("description") or ""))
            elif isinstance(proj, str):
                parts.append(proj)
    return " ".join(p for p in parts if p and p.strip())


def _score_domain(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Tuple[float, str]:
    cv_domain_text = _build_domain_text(cv_data, is_job=False)
    job_domain_text = _build_domain_text(job_data, is_job=True)
    if not cv_domain_text or not job_domain_text:
        return 0.5, "Insufficient domain text on CV or job; neutral score."
    try:
        cv_emb = get_embeddings([cv_domain_text])[0]
        job_emb = get_embeddings([job_domain_text])[0]
        cos = cosine_similarity(cv_emb, job_emb)
        span = DOMAIN_COSINE_CEIL - DOMAIN_COSINE_FLOOR
        score = max(0.0, min(1.0, (cos - DOMAIN_COSINE_FLOOR) / span)) if span > 0 else cos
        return score, f"Embedding cosine of CV domain vs job domain = {cos:.3f} (rescaled to {score:.2f})."
    except Exception as e:
        return 0.5, f"Domain embedding failed ({str(e)[:60]}); neutral score."


def _score_ats_keywords(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Tuple[float, str]:
    """
    Share of the posting's skill terms that appear verbatim (or under a known
    alias) somewhere on the CV. This is what a keyword filter actually checks;
    unlike hard_skills it gives no credit for family or transferable matches.

    The previous BM25 average was min-max normalized across CV sections, so
    the best section always scored 1.0 and the worst 0.0 regardless of fit.
    """
    cv_text = " ".join(_flatten_cv_skill_mentions(cv_data))
    if not cv_text.strip():
        return 0.0, "CV has no extractable section text for ATS scoring."
    required, preferred = _technical_skill_lists(job_data)
    terms: List[str] = []
    seen: set = set()
    for t in required + preferred:
        key = _norm_key(t)
        if key and key not in seen:
            seen.add(key)
            terms.append(t)
    if not terms:
        return 0.5, "Job lists no skill terms to check; neutral score."
    present = [t for t in terms if (p := _skill_mention_pattern(t)) is not None and p.search(cv_text)]
    score = len(present) / len(terms)
    absent = [t for t in terms if t not in present]
    detail = f"; not found verbatim: {', '.join(absent[:6])}" if absent else ""
    return score, f"{len(present)} of {len(terms)} job skill terms appear verbatim on the CV{detail}."


# ============================================
# Main Multi-Dimensional Scorer
# ============================================

def _calculate_compatibility_score_v2_internal(
    cv_data: Dict[str, Any], job_data: Dict[str, Any]
) -> CompatibilityReport:
    matches = _match_skill_pairs_internal(cv_data, job_data)
    eligibility = _check_eligibility(cv_data, job_data)
    job_level = str(job_data.get("job_level") or job_data.get("level") or job_data.get("job_title") or "")

    hard_score, hard_rationale = _score_hard_skills(matches, job_level)
    exp_score, exp_rationale = _score_experience(cv_data, job_data)
    sen_score, sen_rationale = _score_seniority(cv_data, job_data)
    dom_score, dom_rationale = _score_domain(cv_data, job_data)
    ats_score, ats_rationale = _score_ats_keywords(cv_data, job_data)

    dimensions = [
        DimensionScore(name="hard_skills", score=hard_score,
                       weight=COMPATIBILITY_WEIGHTS["hard_skills"], rationale=hard_rationale),
        DimensionScore(name="experience", score=exp_score,
                       weight=COMPATIBILITY_WEIGHTS["experience"], rationale=exp_rationale),
        DimensionScore(name="seniority", score=sen_score,
                       weight=COMPATIBILITY_WEIGHTS["seniority"], rationale=sen_rationale),
        DimensionScore(name="domain", score=dom_score,
                       weight=COMPATIBILITY_WEIGHTS["domain"], rationale=dom_rationale),
        DimensionScore(name="ats_keywords", score=ats_score,
                       weight=COMPATIBILITY_WEIGHTS["ats_keywords"], rationale=ats_rationale),
    ]
    gap = _build_gap_analysis_internal(matches, cv_data, job_data, eligibility)
    return _finish_report(dimensions, eligibility, gap)


def _finish_report(
    dimensions: List[DimensionScore],
    eligibility: List[EligibilityCheck],
    gap: GapAnalysis,
    warnings: Optional[List[str]] = None,
) -> CompatibilityReport:
    """
    Aggregate, level, eligibility cap and interpretation from the skill/
    experience/seniority/domain/ATS dimensions plus the eligibility checks.

    Shared by the full scorer and by rescore_eligibility, which swaps in new
    eligibility checks (after the user answered questions) without re-running
    the embedding and LLM parts of scoring.
    """
    warnings = list(warnings or [])
    dims = [d for d in dimensions if d.name != "eligibility"]
    # Weigh eligibility only when the posting states such requirements and
    # the CV lets us decide at least one; otherwise it would be a free 1.0
    # (no requirements) or a penalty for silence (all unclear).
    elig_score, elig_rationale = _score_eligibility(eligibility)
    if elig_score is not None:
        dims.insert(1, DimensionScore(name="eligibility", score=elig_score,
                                      weight=COMPATIBILITY_WEIGHTS["eligibility"], rationale=elig_rationale))

    weight_total = sum(d.weight for d in dims) or 1.0
    aggregate = max(0.0, min(1.0, sum(d.score * d.weight for d in dims) / weight_total))
    level = _band_for_score(aggregate)

    # A stated must-have the CV positively fails (wrong field of study, missing
    # required language) caps the band: the number can still be high because
    # everything else fits, but 'high compatibility' would mislead the user.
    unmet = [c for c in eligibility if c.status == "unmet"]
    if unmet and level in ("high", "excellent"):
        level = "medium"
        warnings.append(
            "Level capped at medium: the CV does not meet "
            + ", ".join(c.requirement for c in unmet) + "."
        )

    gap = gap.model_copy(update={"eligibility": list(eligibility)})
    scores = {d.name: d.score for d in dims}
    interpretation = _interpret_report(
        level,
        scores.get("hard_skills", 1.0),
        scores.get("experience", 1.0),
        scores.get("seniority", 1.0),
        scores.get("domain", 1.0),
        scores.get("ats_keywords", 1.0),
        gap,
    )

    return CompatibilityReport(
        aggregate_score=round(aggregate, 4),
        level=level,
        interpretation=interpretation,
        dimensions=dims,
        gap_analysis=gap,
        warnings=warnings,
    )


def _interpret_report(
    level: str,
    hard: float,
    exp: float,
    sen: float,
    dom: float,
    ats: float,
    gap: GapAnalysis,
) -> str:
    parts = [f"Overall {level} compatibility."]
    req_missing = [m.required_skill for m in gap.missing_skills if m.is_required]
    pref_missing = [m.required_skill for m in gap.missing_skills if not m.is_required]
    # Covered means held, or held via an interchangeable related skill. A
    # learnable skill is still a gap and is named as one.
    req_covered = sum(1 for m in gap.matched_skills + gap.transferable_skills if m.is_required)
    req_total = req_covered + sum(1 for m in gap.learnable_skills + gap.missing_skills if m.is_required)
    if req_total:
        parts.append(f"Required skills: {req_covered} of {req_total} covered.")
    if req_missing:
        parts.append(f"Genuine gaps on required skills: {', '.join(req_missing)}.")
    if gap.transferable_skills:
        parts.append(f"{len(gap.transferable_skills)} skills are covered via a closely related skill.")
    learnable = [
        f"{m.required_skill} (via {m.matched_with})" if m.matched_with else m.required_skill
        for m in gap.learnable_skills
    ]
    if learnable:
        parts.append(f"Not on the CV but learnable from adjacent experience: {', '.join(learnable)}.")
    if pref_missing:
        parts.append(f"Nice-to-haves not on the CV: {', '.join(pref_missing)}.")
    unmet = [c.requirement for c in gap.eligibility if c.status == "unmet"]
    unclear = [c.requirement for c in gap.eligibility if c.status == "unclear"]
    if unmet:
        parts.append(f"Eligibility not met: {', '.join(unmet)}.")
    if unclear:
        parts.append(f"Not on the CV, so not scored until you confirm: {', '.join(unclear)}.")
    if hard < 0.5:
        parts.append("Hard-skill coverage is the main weakness.")
    if exp < 0.5:
        parts.append("Years-of-experience is below requirement.")
    if sen < 0.5:
        parts.append("Seniority signals are weaker than the job level.")
    return " ".join(parts)


@tool
def calculate_compatibility_score_v2(cv_json: str, job_json: str) -> str:
    """
    Multi-dimensional compatibility score (v2) with skill-graph transferability.

    Combines up to six dimensions:
    - hard_skills (40%): required-skill coverage via direct/family/embedding/LLM
      cascade, plus a level-scaled nice-to-have bonus capped just above it
    - eligibility (15%, only when the posting states any): required languages,
      degree / field of study, enrollment, checked as met / unclear / unmet
    - experience (15%): years-of-experience match with steep under-penalty
    - seniority (10%): ordinal job-level vs candidate title comparison
    - domain (10%): rescaled embedding cosine of domain context
    - ats_keywords (10%): share of job skill terms present verbatim on the CV

    Returns a CompatibilityReport with full per-dimension breakdown and a
    GapAnalysis bucketing skills into matched / transferable / missing /
    over-qualified plus eligibility checks, including draft bridge bullets
    for transferable skills. An unmet eligibility requirement caps the level
    at 'medium' and adds a warning saying so.

    Args:
        cv_json: ResumeInfo JSON from cv_agent.
        job_json: JobRequirements JSON from job_agent.

    Returns:
        JSON-serialized CompatibilityReport.
    """
    try:
        cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
        job_data = json.loads(job_json) if isinstance(job_json, str) else job_json
    except json.JSONDecodeError as e:
        empty = CompatibilityReport(
            aggregate_score=0.0,
            level="unknown",
            interpretation=f"Invalid JSON input: {str(e)}",
            dimensions=[],
            gap_analysis=GapAnalysis(),
            warnings=[f"JSON parse error: {str(e)}"],
        )
        return empty.model_dump_json(indent=2)

    if not isinstance(cv_data, dict) or not isinstance(job_data, dict):
        empty = CompatibilityReport(
            aggregate_score=0.0,
            level="unknown",
            interpretation="CV and job inputs must be JSON objects.",
            dimensions=[],
            gap_analysis=GapAnalysis(),
            warnings=["Non-object inputs"],
        )
        return empty.model_dump_json(indent=2)

    try:
        report = _calculate_compatibility_score_v2_internal(cv_data, job_data)
        return report.model_dump_json(indent=2)
    except Exception as e:
        empty = CompatibilityReport(
            aggregate_score=0.0,
            level="unknown",
            interpretation=f"Unexpected error during scoring: {str(e)}",
            dimensions=[],
            gap_analysis=GapAnalysis(),
            warnings=[str(e)],
        )
        return empty.model_dump_json(indent=2)


# ============================================
# Gap Analysis
# ============================================

def _detect_over_qualified_signals(
    cv_data: Dict[str, Any], job_data: Dict[str, Any], matches: List[SkillMatch]
) -> List[str]:
    """
    Surface candidate signals that exceed the job's stated needs (large team
    leadership for an IC role, advanced architecture for a junior role, etc.).
    Heuristic only - full LLM-judged version can replace this later.
    """
    signals: List[str] = []
    job_level = (job_data.get("job_level") or "").lower()
    if not job_level:
        return signals

    cand_scores: List[int] = []
    experience = cv_data.get("experience") or cv_data.get("experiences") or []
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, dict):
                pos = exp.get("position") or exp.get("title")
                if pos:
                    s = _seniority_to_score(str(pos))
                    if s is not None:
                        cand_scores.append(s)
    job_score = _seniority_to_score(job_level)
    if job_score is not None and cand_scores and max(cand_scores) - job_score >= 2:
        signals.append(
            f"Candidate has held titles that score {max(cand_scores) - job_score} levels above the '{job_level}' role."
        )

    leadership_terms = ["led team", "managed team", "architect", "directed", "mentored", "head of"]
    blobs = " ".join(_flatten_cv_skill_mentions(cv_data)).lower()
    found = [t for t in leadership_terms if t in blobs]
    if found and job_score is not None and job_score <= 3:
        signals.append(
            f"Leadership/architecture signals on CV ({', '.join(found[:3])}) exceed typical scope for this level."
        )

    return signals


def _build_gap_analysis_internal(
    matches: List[SkillMatch],
    cv_data: Dict[str, Any],
    job_data: Dict[str, Any],
    eligibility: Optional[List[EligibilityCheck]] = None,
) -> GapAnalysis:
    buckets: Dict[str, List[SkillMatch]] = {"matched": [], "transferable": [], "learnable": [], "missing": []}
    for m in matches:
        buckets[_bucket(m)].append(m)
    over = _detect_over_qualified_signals(cv_data, job_data, matches)
    if eligibility is None:
        eligibility = _check_eligibility(cv_data, job_data)
    return GapAnalysis(
        matched_skills=buckets["matched"],
        transferable_skills=buckets["transferable"],
        learnable_skills=buckets["learnable"],
        missing_skills=buckets["missing"],
        over_qualified_signals=over,
        eligibility=eligibility,
    )


@tool
def build_gap_analysis(skill_matches_json: str, cv_json: str, job_json: str) -> str:
    """
    Bucket a list of SkillMatch entries into matched / transferable / missing
    and surface over-qualification signals.

    Args:
        skill_matches_json: JSON list of SkillMatch dicts (from match_skill_pairs).
        cv_json: ResumeInfo JSON.
        job_json: JobRequirements JSON.

    Returns:
        JSON-serialized GapAnalysis.
    """
    try:
        raw_matches = json.loads(skill_matches_json) if isinstance(skill_matches_json, str) else skill_matches_json
        cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
        job_data = json.loads(job_json) if isinstance(job_json, str) else job_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})

    matches: List[SkillMatch] = []
    if isinstance(raw_matches, list):
        for item in raw_matches:
            try:
                matches.append(SkillMatch.model_validate(item))
            except Exception:
                continue

    gap = _build_gap_analysis_internal(matches, cv_data, job_data)
    return gap.model_dump_json(indent=2)


# ============================================
# Legacy Compatibility Score (v1 shim)
# ============================================

@tool
def calculate_compatibility_score(cv_json: str, job_json: str, bm25_weight: float = 0.4, semantic_weight: float = 0.6) -> str:
    """
    Legacy compatibility score (v1) - kept for backwards compatibility.

    Internally delegates to calculate_compatibility_score_v2 (multi-dimensional
    scoring with skill-graph transferability) and projects the result back into
    the v1 response shape so existing consumers continue to work.

    The v2 report is also embedded under the `report_v2` key for callers that
    want to opt into the richer breakdown.

    Args:
        cv_json: ResumeInfo JSON string from cv_agent
        job_json: JobRequirements JSON string from job_agent
        bm25_weight: Retained only for backwards compatibility (v2 uses fixed
            weights via COMPATIBILITY_WEIGHTS).
        semantic_weight: Retained only for backwards compatibility.

    Returns:
        JSON string with compatibility_score, level, interpretation, breakdown
        (bm25_score, semantic_similarity, bm25_weight, semantic_weight), and
        report_v2 (the full CompatibilityReport).
    """
    error_details: List[str] = []

    try:
        try:
            cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
        except json.JSONDecodeError as e:
            error_details.append(f"CV JSON parse error: {str(e)}")
            cv_data = {}

        try:
            job_data = json.loads(job_json) if isinstance(job_json, str) else job_json
        except json.JSONDecodeError as e:
            error_details.append(f"Job JSON parse error: {str(e)}")
            job_data = {}

        if not isinstance(cv_data, dict) or not isinstance(job_data, dict):
            return json.dumps({
                "error": "CV and job inputs must be JSON objects",
                "error_details": error_details,
                "compatibility_score": 0.0,
                "level": "unknown",
                "interpretation": "Cannot calculate compatibility: malformed input.",
            })

        report = _calculate_compatibility_score_v2_internal(cv_data, job_data)

        ats_score = next(
            (d.score for d in report.dimensions if d.name == "ats_keywords"), 0.0
        )
        domain_score = next(
            (d.score for d in report.dimensions if d.name == "domain"), 0.0
        )

        v1_level_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "excellent": "high",
            "unknown": "unknown",
        }
        legacy_level = v1_level_map.get(report.level, report.level)

        result = {
            "compatibility_score": round(report.aggregate_score, 4),
            "level": legacy_level,
            "interpretation": report.interpretation,
            "breakdown": {
                "bm25_score": round(ats_score, 4),
                "semantic_similarity": round(domain_score, 4),
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "scoring_version": "v2_via_legacy_shim",
            },
            "report_v2": json.loads(report.model_dump_json()),
        }

        if error_details or report.warnings:
            result["warnings"] = error_details + report.warnings

        return json.dumps(result)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_details": error_details + [f"Traceback: {error_trace[:500]}"],
            "compatibility_score": 0.0,
            "level": "unknown",
            "interpretation": "An unexpected error occurred while calculating compatibility. Please try again or use manual analysis."
        })
