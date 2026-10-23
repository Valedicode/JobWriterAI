"""The teacher: the app's extraction and scoring, called as a black box (prior work, NLP_PROJECT_PLAN.md §1).

Two things are added around it without changing the app:

- The persistent caches (ml.labeling.caches) for extractions and embeddings.
- A record of the API errors the scorer swallows. scoring_agent still returns a score when a call
  fails: a failed skill embedding becomes "no semantic match", a failed judge call leaves skills
  unjudged, a failed domain embedding gives a neutral 0.5. Such a score would be wrong without
  saying so, so the labelling run retries it instead of keeping it.
"""
import threading

from langchain_core.callbacks import BaseCallbackHandler

from app.agents import scoring_agent
from app.agents.job_agent import retrieve_text
from app.models.schemas import JobRequirements
from ml.labeling.caches import EmbeddingCache
from ml.labeling.resume_text_extractor import extract_resume

# The rationale scoring_agent gives a skill nothing resolved. After the judge step it means the
# batched judge call failed or left the skill out.
UNJUDGED = "No direct, family, or semantic match found in candidate skills."
# Errors a retry can't fix: a bad key, no credit left, a provider refusing the request's region, and
# OpenRouter finding no endpoint for the model (e.g. the pinned provider is excluded by the account's
# privacy settings).
FATAL_STATUS = {401, 402, 403, 404}

_ORIGINAL_GET_EMBEDDINGS = scoring_agent.get_embeddings
_ORIGINAL_CHAT_MODEL = scoring_agent.chat_model
_local = threading.local()


class FatalAPIError(RuntimeError):
    """An API error that stops the whole run (see FATAL_STATUS)."""


def is_fatal(error: BaseException) -> bool:
    return getattr(error, "status_code", None) in FATAL_STATUS


def swallowed_errors() -> list[BaseException]:
    """The API errors recorded on this thread since the last ``score`` call started."""
    if not hasattr(_local, "errors"):
        _local.errors = []
    return _local.errors


class _RecordErrors(BaseCallbackHandler):
    def __init__(self, errors: list[BaseException]):
        self.errors = errors

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        self.errors.append(error)


def _recording_chat_model(role, **kwargs):
    model = _ORIGINAL_CHAT_MODEL(role, **kwargs)
    model.callbacks = [_RecordErrors(swallowed_errors())]
    return model


def install(embeddings: EmbeddingCache | None) -> None:
    """Route the scorer's embeddings through ``embeddings`` (None: no persistent cache) and record
    the errors it swallows. Affects this process only."""
    fetch = embeddings.wrap(_ORIGINAL_GET_EMBEDDINGS) if embeddings else _ORIGINAL_GET_EMBEDDINGS

    def get_embeddings(texts, model="text-embedding-3-small"):
        try:
            return fetch(texts, model=model)
        except Exception as e:
            swallowed_errors().append(e)
            raise

    scoring_agent.get_embeddings = get_embeddings
    scoring_agent.chat_model = _recording_chat_model
    scoring_agent._get_single_embedding_cached.cache_clear()


def extract(kind: str, text: str) -> dict:
    """The teacher's extraction of one resume or job text (raises on a failed call)."""
    if kind == "resume":
        result = extract_resume(text)
        return {"cv": result.cv, "notes": result.notes}
    if kind == "job":
        return {"job": JobRequirements.model_validate_json(retrieve_text.invoke({"info": text})).model_dump()}
    raise ValueError(f"unknown kind {kind!r}")


def is_degenerate(cv: dict) -> bool:
    """An extraction with neither skills nor experience: nothing to score."""
    return not cv.get("skills") and not cv.get("experience")


def score(cv: dict, job: dict) -> tuple[dict, list[str], list[BaseException]]:
    """The teacher's report for one pair, its flags, and the API errors the scorer swallowed."""
    errors = swallowed_errors()
    errors.clear()
    report = scoring_agent._calculate_compatibility_score_v2_internal(cv, job)
    gap = report.gap_analysis
    flags = []
    if any(m.rationale == UNJUDGED for m in gap.matched_skills + gap.transferable_skills + gap.missing_skills):
        flags.append("llm_transferability_unjudged")
    if any(d.rationale.startswith("Domain embedding failed") for d in report.dimensions):
        flags.append("domain_embedding_failed")
    return report.model_dump(), flags, list(errors)
