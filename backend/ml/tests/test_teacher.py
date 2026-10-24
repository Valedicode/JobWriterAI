"""The teacher wrapper records the API errors the scorer swallows, so a wrong score isn't kept.

Run from backend/:  python -m pytest ml/tests
"""
import pytest
from langchain_core.language_models import BaseChatModel

from app.models.schemas import CompatibilityReport, GapAnalysis
from ml.labeling import teacher


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FailingChat(BaseChatModel):
    status_code: int = 429

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise StatusError(self.status_code)

    @property
    def _llm_type(self) -> str:
        return "failing"


@pytest.fixture
def restore_scorer(monkeypatch):
    """install() patches scoring_agent; put its functions back after the test."""
    agent = teacher.scoring_agent
    monkeypatch.setattr(agent, "get_embeddings", agent.get_embeddings)
    monkeypatch.setattr(agent, "chat_model", agent.chat_model)


def _failing_embeddings(texts, model="m"):
    raise StatusError(500)


def test_a_failed_embedding_is_recorded_and_still_raised(monkeypatch, restore_scorer):
    monkeypatch.setattr(teacher, "_ORIGINAL_GET_EMBEDDINGS", _failing_embeddings)
    teacher.install(None)
    teacher.swallowed_errors().clear()
    with pytest.raises(StatusError):
        teacher.scoring_agent.get_embeddings(["python"])
    assert [e.status_code for e in teacher.swallowed_errors()] == [500]


def test_a_failed_judge_call_is_recorded(monkeypatch, restore_scorer):
    monkeypatch.setattr(teacher, "_ORIGINAL_CHAT_MODEL", lambda role, **kw: _FailingChat())
    teacher.install(None)
    teacher.swallowed_errors().clear()
    with pytest.raises(StatusError):
        teacher.scoring_agent.chat_model("judge", temperature=0).invoke("verdicts?")
    assert [e.status_code for e in teacher.swallowed_errors()] == [429]


def test_score_returns_the_errors_the_scorer_swallowed(monkeypatch, restore_scorer):
    monkeypatch.setattr(teacher, "_ORIGINAL_GET_EMBEDDINGS", _failing_embeddings)
    teacher.install(None)

    def scorer_that_swallows(cv, job):
        try:
            teacher.scoring_agent.get_embeddings(["python"])
        except Exception:
            pass  # like _embedding_transferability: the failure becomes "no match"
        return CompatibilityReport(aggregate_score=0.3, level="low", dimensions=[], gap_analysis=GapAnalysis())

    monkeypatch.setattr(teacher.scoring_agent, "_calculate_compatibility_score_v2_internal", scorer_that_swallows)
    report, flags, errors = teacher.score({"skills": ["Go"]}, {})
    assert report["aggregate_score"] == 0.3 and flags == []
    assert [e.status_code for e in errors] == [500]
    # The next score starts with a clean record.
    monkeypatch.setattr(teacher.scoring_agent, "_calculate_compatibility_score_v2_internal",
                        lambda cv, job: CompatibilityReport(aggregate_score=0.5, level="medium", dimensions=[],
                                                            gap_analysis=GapAnalysis()))
    assert teacher.score({"skills": ["Go"]}, {})[2] == []


def test_fatal_statuses():
    assert all(teacher.is_fatal(StatusError(code)) for code in (401, 402, 403, 404))
    assert not teacher.is_fatal(StatusError(429)) and not teacher.is_fatal(ValueError("x"))
