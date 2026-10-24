"""The retest compares two runs of the same teacher on the rows clean in both.

Run from backend/:  python -m pytest ml/tests
"""
import json

import pytest

from ml.labeling import teacher_retest as rt

META = {"tag": "teacher-v1", "commit": "abc", "dirty": False, "route": "openrouter", "providers": ["relace"],
        "models": {"extract": "m-1", "judge": "m-1"}, "embedding_route": "openai",
        "embedding_model": "text-embedding-3-small", "reasoning": False,
        "dataset": {"id": "d", "revision": "r", "split": "train"}, "cache": False}


def _row(i, score, level="low", status="ok", flags=()):
    return {"row_id": i, "status": status, "flags": list(flags),
            "report": {"aggregate_score": score, "level": level,
                       "dimensions": [{"name": "hard_skills", "score": score}, {"name": "domain", "score": 0.5}]}}


def _run(tmp_path, name, rows, meta=META, extractions=()):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    path.with_suffix(".teacher_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    path.with_suffix(".extractions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in extractions),
                                                      encoding="utf-8")
    return path


def test_icc_counts_a_shift_that_spearman_ignores():
    assert rt.icc_a1([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert rt.icc_a1([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 3)  # MSR 2, MSC 1.5, MSE 0
    assert rt.agreement([1, 2, 3], [2, 3, 4])["spearman"] == 1.0


def test_only_rows_clean_in_both_runs_are_compared(tmp_path):
    a = _run(tmp_path, "a.jsonl", [_row(1, 0.2), _row(2, 0.5), _row(3, 0.7, "medium"), _row(4, 0.4),
                                   _row(5, 0.3, flags=["domain_embedding_failed"])],
             extractions=[{"kind": "resume", "sha256": "r1", "status": "ok", "value": {"x": 1}},
                          {"kind": "job", "sha256": "j1", "status": "ok", "value": {"y": 1}}])
    b = _run(tmp_path, "b.jsonl", [_row(1, 0.25), _row(2, 0.45), _row(3, 0.7, "high"), _row(4, 0.4, status="error"),
                                   _row(5, 0.3)],
             extractions=[{"kind": "resume", "sha256": "r1", "status": "ok", "value": {"x": 2}},
                          {"kind": "job", "sha256": "j1", "status": "ok", "value": {"y": 1}}])
    s = rt.retest(a, b)
    assert s["rows"] == {"a": 4, "b": 4, "compared": 3}
    assert s["agreement"]["aggregate"]["spearman"] == 1.0
    assert s["agreement"]["aggregate"]["max_abs_diff"] == pytest.approx(0.05)
    assert set(s["agreement"]) == {"aggregate", "hard_skills", "domain"}
    assert s["level"] == {"same": round(2 / 3, 3), "off_by_more_than_one": 0}
    assert s["identical_extractions"] == {"resume": {"texts": 1, "identical": 0.0},
                                          "job": {"texts": 1, "identical": 1.0}}
    assert s["warning"] is None


def test_runs_from_different_teachers_are_refused(tmp_path):
    rows = [_row(i, 0.1 * i) for i in range(1, 5)]
    a = _run(tmp_path, "a.jsonl", rows)
    b = _run(tmp_path, "b.jsonl", rows, meta={**META, "providers": ["together"]})
    with pytest.raises(SystemExit, match="providers"):
        rt.retest(a, b)


def test_a_cached_run_is_warned_about(tmp_path):
    rows = [_row(i, 0.1 * i) for i in range(1, 5)]
    s = rt.retest(_run(tmp_path, "a.jsonl", rows, meta={**META, "cache": True}), _run(tmp_path, "b.jsonl", rows))
    assert "a.jsonl used the teacher cache" in s["warning"]
