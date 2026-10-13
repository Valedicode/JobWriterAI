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
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

from app.models.schemas import (
    SkillMatch,
    DimensionScore,
    GapAnalysis,
    CompatibilityReport,
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
    "experience": 0.15,
    "seniority": 0.15,
    "domain": 0.15,
    "ats_keywords": 0.15,
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
    skills = cv_data.get("skills") or []
    if isinstance(skills, list):
        mentions.extend(str(s) for s in skills if s)
    elif isinstance(skills, str):
        mentions.append(skills)

    experience = cv_data.get("experience") or cv_data.get("experiences") or []
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, str):
                mentions.append(exp)
            elif isinstance(exp, dict):
                parts = [
                    str(exp.get("position", "")),
                    str(exp.get("company", "")),
                ]
                resp = exp.get("responsibilities")
                if isinstance(resp, list):
                    parts.extend(str(r) for r in resp if r)
                elif isinstance(resp, str):
                    parts.append(resp)
                mentions.append(" ".join(p for p in parts if p))
    elif isinstance(experience, str):
        mentions.append(experience)

    projects = cv_data.get("projects") or []
    if isinstance(projects, list):
        for proj in projects:
            if isinstance(proj, str):
                mentions.append(proj)
            elif isinstance(proj, dict):
                parts = [str(proj.get("name", "")), str(proj.get("description", ""))]
                mentions.append(" ".join(p for p in parts if p))
    elif isinstance(projects, str):
        mentions.append(projects)

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
    norm_req = skill_graph_service.normalize(required_skill)
    if not norm_req:
        return None

    declared = _candidate_skills_normalized(cv_data)
    for cand in declared:
        if skill_graph_service.is_direct_match(required_skill, cand):
            return cand

    pattern = re.compile(rf"\b{re.escape(norm_req)}\b", re.IGNORECASE)
    for blob in _flatten_cv_skill_mentions(cv_data):
        if pattern.search(blob.lower()):
            return required_skill
    return None


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

class _LLMTransferabilityItem(BaseModel):
    """Schema for a single LLM-judged transferability assessment."""
    required_skill: str = Field(description="The required skill being assessed")
    best_candidate_skill: Optional[str] = Field(
        default=None,
        description="Best candidate skill that could bridge to the required skill, if any"
    )
    transferability: float = Field(
        ge=0.0, le=1.0,
        description="0-1 score for how well candidate skills cover this requirement"
    )
    rationale: str = Field(description="Short explanation of the transferability judgment")
    bridge_bullet: Optional[str] = Field(
        default=None,
        description="Suggested resume bullet phrasing that surfaces the bridge skill (only when transferability >= 0.55)"
    )


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

For each required skill, decide how well the candidate's declared skills and
experience cover it. Think in terms of underlying paradigms and learning curve,
not surface keywords:

- Frameworks in the same family (e.g. React/Vue/Angular, PyTorch/TensorFlow,
  Postgres/MySQL) are highly transferable - rate 0.80-0.92.
- Cross-paradigm but adjacent (e.g. Java -> Kotlin, Python -> Ruby,
  REST -> GraphQL) - rate 0.55-0.75.
- Same general domain but meaningful learning curve (e.g. frontend -> backend
  in same language) - rate 0.35-0.55.
- Genuinely missing or unrelated - rate 0.0-0.30.

Score 1.0 only if the candidate already lists the exact skill.

For each item return:
- required_skill: echo the input
- best_candidate_skill: the single most relevant candidate skill (or null)
- transferability: 0-1 (calibrated as above)
- rationale: 1-2 sentences explaining the bridge or why it's missing
- bridge_bullet: ONLY when transferability >= 0.55, draft a single resume bullet
  (<= 25 words, action-verb led, no fabrication) that surfaces the candidate's
  existing experience in language that demonstrates readiness for the required
  skill. Use only facts implied by the candidate's declared skills/experience.
  Otherwise return null.
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
            "cv_summary": cv_summary_text[:2000],
            "required": "\n".join(f"- {s}" for s in unresolved_required),
        })
        return {item.required_skill: item for item in result.assessments}
    except Exception:
        return {}


# ============================================
# Skill Matching Cascade
# ============================================

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


def _match_skill_pairs_internal(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[SkillMatch]:
    """
    Resolve every required and preferred job skill against the candidate's CV
    using the cascade: direct -> family -> embedding -> LLM.
    """
    required_raw = job_data.get("required_skills") or job_data.get("skills") or []
    preferred_raw = job_data.get("preferred_skills") or []

    required_list = [str(s).strip() for s in required_raw if s and str(s).strip()] if isinstance(required_raw, list) else []
    preferred_list = [str(s).strip() for s in preferred_raw if s and str(s).strip()] if isinstance(preferred_raw, list) else []

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
                judged = llm_results.get(m.required_skill)
                if judged is None:
                    continue
                if judged.transferability >= TRANSFERABLE_THRESHOLD:
                    matches[idx] = SkillMatch(
                        required_skill=m.required_skill,
                        matched_with=judged.best_candidate_skill,
                        kind="transferable",
                        transferability=float(judged.transferability),
                        rationale=judged.rationale,
                        bridge_bullet=judged.bridge_bullet,
                        is_required=m.is_required,
                    )
                else:
                    matches[idx] = SkillMatch(
                        required_skill=m.required_skill,
                        matched_with=judged.best_candidate_skill,
                        kind="missing",
                        transferability=float(judged.transferability),
                        rationale=judged.rationale,
                        bridge_bullet=None,
                        is_required=m.is_required,
                    )

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
        transferability (0-1), rationale, and bridge_bullet (when applicable).
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
# Dimension Scorers
# ============================================

def _score_hard_skills(matches: List[SkillMatch]) -> Tuple[float, str]:
    """
    Weighted average of transferability across required + preferred skills.
    Required skills are weighted 2x preferred.
    """
    if not matches:
        return 0.0, "No required or preferred skills declared on the job."
    total = 0.0
    weight_sum = 0.0
    for m in matches:
        w = 2.0 if m.is_required else 1.0
        total += m.transferability * w
        weight_sum += w
    score = total / weight_sum if weight_sum > 0 else 0.0
    matched_count = sum(1 for m in matches if m.kind in ("direct", "family"))
    transferable_count = sum(1 for m in matches if m.kind == "transferable")
    missing_count = sum(1 for m in matches if m.kind == "missing")
    rationale = (
        f"{matched_count} matched, {transferable_count} transferable, "
        f"{missing_count} missing across {len(matches)} job skills "
        "(required weighted 2x preferred)."
    )
    return score, rationale


_YEARS_RE = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:years?|yrs?)", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(
    r"(\d{4})\s*[-–to]+\s*(present|current|now|\d{4})", re.IGNORECASE
)


def _estimate_candidate_years(cv_data: Dict[str, Any]) -> float:
    """
    Rough heuristic for candidate's years of experience from CV text.

    Looks for explicit "X years" mentions first, then falls back to summing
    YYYY-YYYY (or YYYY-Present) ranges in experience entries.
    """
    blobs = _flatten_cv_skill_mentions(cv_data)
    full_text = " ".join(blobs)

    explicit = [int(m.group(1)) for m in _YEARS_RE.finditer(full_text)]
    if explicit:
        return float(max(explicit))

    total_years = 0.0
    current_year = datetime.now().year
    for match in _DATE_RANGE_RE.finditer(full_text):
        try:
            start = int(match.group(1))
            end_token = match.group(2).lower()
            end = current_year if end_token in ("present", "current", "now") else int(end_token)
            if start <= end <= current_year and end - start <= 50:
                total_years += end - start
        except (ValueError, AttributeError):
            continue
    return total_years


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


_SENIORITY_SCORE = {
    "intern": 0,
    "entry": 1,
    "junior": 1,
    "associate": 2,
    "mid": 3,
    "intermediate": 3,
    "senior": 4,
    "staff": 5,
    "principal": 6,
    "lead": 5,
    "manager": 5,
    "director": 6,
    "head": 6,
    "vp": 7,
    "cto": 7,
}


def _seniority_to_score(level_text: str) -> Optional[int]:
    if not level_text:
        return None
    txt = level_text.lower()
    for key, val in _SENIORITY_SCORE.items():
        if key in txt:
            return val
    return None


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
        score = cosine_similarity(cv_emb, job_emb)
        return score, f"Embedding cosine of CV domain vs job domain = {score:.3f}."
    except Exception as e:
        return 0.5, f"Domain embedding failed ({str(e)[:60]}); neutral score."


def _score_ats_keywords(cv_data: Dict[str, Any], job_data: Dict[str, Any]) -> Tuple[float, str]:
    """Reuse the existing BM25 average; ATS systems still do exact-keyword filtering."""
    cv_sections = _flatten_cv_skill_mentions(cv_data)
    if not cv_sections:
        return 0.0, "CV has no extractable section text for ATS scoring."
    job_text_parts = []
    job_text_parts.append(str(job_data.get("job_title") or ""))
    for r in (job_data.get("responsibilities") or []):
        if r:
            job_text_parts.append(str(r))
    for s in (job_data.get("required_skills") or []):
        if s:
            job_text_parts.append(str(s))
    for q in (job_data.get("qualifications") or []):
        if q:
            job_text_parts.append(str(q))
    job_text = " ".join(p for p in job_text_parts if p.strip())
    if not job_text:
        return 0.5, "Job has no ATS keyword text; neutral score."
    try:
        scores = _calculate_bm25_scores(job_text, cv_sections)
        if not scores:
            return 0.0, "BM25 produced no scores."
        avg = sum(scores) / len(scores)
        return avg, f"BM25 average across {len(cv_sections)} CV sections = {avg:.3f}."
    except Exception as e:
        return 0.0, f"BM25 calculation failed: {str(e)[:60]}."


# ============================================
# Main Multi-Dimensional Scorer
# ============================================

def _calculate_compatibility_score_v2_internal(
    cv_data: Dict[str, Any], job_data: Dict[str, Any]
) -> CompatibilityReport:
    warnings: List[str] = []

    matches = _match_skill_pairs_internal(cv_data, job_data)

    hard_score, hard_rationale = _score_hard_skills(matches)
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

    weight_total = sum(d.weight for d in dimensions) or 1.0
    aggregate = sum(d.score * d.weight for d in dimensions) / weight_total
    aggregate = max(0.0, min(1.0, aggregate))

    level = _band_for_score(aggregate)

    gap = _build_gap_analysis_internal(matches, cv_data, job_data)

    interpretation = _interpret_report(level, hard_score, exp_score, sen_score, dom_score, ats_score, gap)

    return CompatibilityReport(
        aggregate_score=round(aggregate, 4),
        level=level,
        interpretation=interpretation,
        dimensions=dimensions,
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
    if gap.matched_skills:
        parts.append(f"{len(gap.matched_skills)} skills directly matched.")
    if gap.transferable_skills:
        parts.append(f"{len(gap.transferable_skills)} transferable skills can be bridged.")
    if gap.missing_skills:
        parts.append(f"{len(gap.missing_skills)} required skills are genuine gaps.")
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

    Combines five dimensions:
    - hard_skills (40%): per-skill transferability via direct/family/embedding/LLM cascade
    - experience (15%): years-of-experience match with steep under-penalty
    - seniority (15%): ordinal job-level vs candidate title comparison
    - domain (15%): embedding cosine of domain context (titles/responsibilities)
    - ats_keywords (15%): BM25 average for exact-keyword ATS filters

    Returns a CompatibilityReport with full per-dimension breakdown and a
    GapAnalysis bucketing skills into matched / transferable / missing /
    over-qualified, including draft bridge bullets for transferable skills.

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
    matches: List[SkillMatch], cv_data: Dict[str, Any], job_data: Dict[str, Any]
) -> GapAnalysis:
    matched: List[SkillMatch] = []
    transferable: List[SkillMatch] = []
    missing: List[SkillMatch] = []
    for m in matches:
        if m.transferability >= MATCHED_THRESHOLD:
            matched.append(m)
        elif m.transferability >= TRANSFERABLE_THRESHOLD:
            transferable.append(m)
        else:
            missing.append(m)
    over = _detect_over_qualified_signals(cv_data, job_data, matches)
    return GapAnalysis(
        matched_skills=matched,
        transferable_skills=transferable,
        missing_skills=missing,
        over_qualified_signals=over,
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
