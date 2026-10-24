"""The labelling run retries scores reached despite a failed call, stops on fatal errors, and
summarizes the latest record of each row.

Run from backend/:  python -m pytest ml/tests
"""
import json

import numpy as np
import pytest

from ml.labeling import generate_teacher_labels as gen
from ml.labeling import teacher
from ml.labeling.caches import ExtractionCache, Store


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _report(score=0.4, level="low"):
    return {"aggregate_score": score, "level": level,
            "dimensions": [{"name": "hard_skills", "score": score}, {"name": "domain", "score": 0.5}]}


@pytest.fixture
def cache():
    c = ExtractionCache(Store(None, "extractions"), "v")
    c.put("resume", "R", {"cv": {"skills": ["Go"]}, "notes": []})
    c.put("resume", "EMPTY", {"cv": {"skills": [], "experience": []}, "notes": []})
    c.put("job", "J", {"job": {"title": "Dev"}})
    return c


def _scores(monkeypatch, *results):
    """teacher.score returns these (report, flags, errors) in turn; returns the call counter."""
    calls = []

    def fake(cv, job):
        calls.append(1)
        return results[min(len(calls), len(results)) - 1]
    monkeypatch.setattr(teacher, "score", fake)
    return calls


def test_rows_follow_the_v0_pilot_order_without_repeated_pairs():
    resumes, jobs = ["r0", "r1", "r0", "r2", "r3"], ["j0", "j1", "j0", "j2", "j3"]  # row 2 repeats row 0
    order = [int(i) for i in np.random.default_rng(42).permutation(5)]
    assert gen.select_rows(resumes, jobs, limit=5) == [i for i in order if i != 2]
    assert gen.select_rows(resumes, jobs, limit=0) == [0, 1, 3, 4]


def test_a_clean_score_is_kept_at_once(monkeypatch, cache):
    calls = _scores(monkeypatch, (_report(), [], []))
    rec = gen.score_row(7, "R", "J", cache, None, attempts=3, backoff_s=0)
    assert (rec["status"], rec["attempts"], rec["flags"], len(calls)) == ("ok", 1, [], 1)
    assert rec["report"]["aggregate_score"] == 0.4 and rec["cost_usd"] is None


def test_a_score_after_a_swallowed_error_is_retried(monkeypatch, cache):
    calls = _scores(monkeypatch, (_report(0.1), [], [StatusError(500)]), (_report(0.4), [], []))
    rec = gen.score_row(7, "R", "J", cache, None, attempts=3, backoff_s=0)
    assert (rec["status"], rec["attempts"], len(calls)) == ("ok", 2, 2)
    assert rec["report"]["aggregate_score"] == 0.4  # the score reached despite the error is dropped


def test_a_row_that_keeps_failing_is_an_error_for_the_next_run(monkeypatch, cache):
    _scores(monkeypatch, (_report(0.1), ["domain_embedding_failed"], [StatusError(500)]))
    rec = gen.score_row(7, "R", "J", cache, None, attempts=3, backoff_s=0)
    assert (rec["status"], rec["attempts"]) == ("error", 3)
    assert "report" not in rec and "HTTP 500" in rec["error"]


def test_a_flag_without_an_api_error_is_kept_after_the_retries(monkeypatch, cache):
    _scores(monkeypatch, (_report(), ["llm_transferability_unjudged"], []))
    rec = gen.score_row(7, "R", "J", cache, None, attempts=3, backoff_s=0)
    assert (rec["status"], rec["attempts"], rec["flags"]) == ("ok", 3, ["llm_transferability_unjudged"])


def test_no_credit_stops_the_run(monkeypatch, cache):
    calls = _scores(monkeypatch, (_report(), [], [StatusError(402)]))
    with pytest.raises(teacher.FatalAPIError):
        gen.score_row(7, "R", "J", cache, None, attempts=3, backoff_s=0)
    assert len(calls) == 1


def test_rows_without_a_usable_extraction_are_not_scored(monkeypatch, cache):
    calls = _scores(monkeypatch, (_report(), [], []))
    assert gen.score_row(1, "EMPTY", "J", cache, None, backoff_s=0)["status"] == "degenerate_cv"
    missing = gen.score_row(2, "unextracted", "J", cache, None, backoff_s=0)
    assert missing["status"] == "error" and "extraction failed" in missing["error"]
    assert calls == []


def test_retries_wait_on_transient_errors_but_not_on_fatal_ones():
    attempts = iter([StatusError(429), "ok"])

    def flaky():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result
    assert gen.with_retries(flaky, attempts=3, backoff_s=0) == ("ok", 2)
    with pytest.raises(teacher.FatalAPIError):
        gen.with_retries(lambda: (_ for _ in ()).throw(StatusError(401)), attempts=3, backoff_s=0)


def test_a_failed_extraction_is_logged_not_cached(monkeypatch, cache):
    def fail(kind, text):
        raise ValueError("unparseable extraction")
    monkeypatch.setattr(teacher, "extract", fail)
    rec = gen.extract_text("job", "NEW", cache, None, attempts=2, backoff_s=0)
    assert (rec["status"], rec["attempts"]) == ("error", 2) and cache.get("job", "NEW") is None
    monkeypatch.setattr(teacher, "extract", lambda kind, text: {"job": {"title": "Ops"}})
    assert gen.extract_text("job", "NEW", cache, None, backoff_s=0)["status"] == "ok"
    assert cache.get("job", "NEW") == {"job": {"title": "Ops"}}


def test_many_failures_in_a_row_stop_the_run(monkeypatch):
    monkeypatch.setattr(gen, "MAX_ERRORS_IN_A_ROW", 3)
    written = []
    tasks = [lambda: {"status": "error", "error": "HTTP 500"}] * 10
    with pytest.raises(gen.StopRun):
        gen.run_parallel(tasks, workers=1, write=written.append, describe=str)
    assert len(written) == 3


def test_spearman_shares_ranks_between_ties():
    assert round(gen.spearman([1, 2, 2, 3], [1, 3, 2, 4]), 4) == 0.9487  # sqrt(0.9)


def test_summary_counts_the_latest_record_and_every_attempts_spend(tmp_path):
    out = tmp_path / "run.jsonl"
    recs = [
        {"row_id": 0, "status": "error", "error": "HTTP 500", "attempts": 4, "cost_usd": 0.001, "latency_s": 9.0},
        {"row_id": 0, "status": "ok", "report": _report(0.4), "flags": [], "attempts": 1, "cost_usd": 0.002, "latency_s": 3.0},
        {"row_id": 1, "status": "ok", "report": _report(0.6, "medium"), "flags": [], "attempts": 2, "cost_usd": 0.002, "latency_s": 5.0},
        {"row_id": 2, "status": "degenerate_cv", "attempts": 0, "latency_s": 0.0},
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    out.with_suffix(".extractions.jsonl").write_text(
        json.dumps({"kind": "job", "sha256": "x", "status": "ok", "cost_usd": 0.0005}) + "\n", encoding="utf-8")
    s = gen.summarize(out, "train", labels=["No Fit", "Good Fit", "Potential Fit"])
    assert s["status"] == {"ok": 2, "degenerate_cv": 1, "error": 0}
    assert s["retried_rows"] == 1
    assert s["spend_usd"] == {"scoring": 0.005, "extraction": 0.0005, "extracted_texts": 1}
    assert s["level_distribution"]["medium"] == 1
    assert s["dimension_means"] == {"hard_skills": 0.5, "domain": 0.5}
    assert s["train_labels"] == {"spearman_aggregate_vs_label": 1.0,
                                 "mean_aggregate_by_label": {"No Fit": 0.4, "Good Fit": 0.6}}
