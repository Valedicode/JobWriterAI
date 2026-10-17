"""The judge comparison measures how far two scores of the same pair are apart.

Run from backend/:  python -m pytest ml/tests
"""
from ml.labeling.compare_judge_models import summarize


def _rec(judge, row_id, run, score, level="low", kinds=None, unjudged=0):
    return {"judge": judge, "row_id": row_id, "run": run, "status": "ok", "score": score, "level": level,
            "kinds": kinds or {"Python": "direct"}, "unjudged": unjudged, "cost_usd": 0.001, "latency_s": 2.0}


def test_two_runs_of_a_pair_are_compared():
    records = [
        _rec("a", 1, 1, 0.40), _rec("a", 1, 2, 0.46, level="medium"),
        _rec("a", 2, 1, 0.30, kinds={"Go": "missing", "SQL": "direct"}),
        _rec("a", 2, 2, 0.30, kinds={"Go": "transferable", "SQL": "direct"}, unjudged=1),
        {"judge": "a", "row_id": 3, "run": 1, "status": "error", "error": "BadRequestError: x", "latency_s": 1.0},
    ]
    s = summarize(records, ["a"], full_rows=1000)["a"]
    assert (s["scores"], s["failed"]) == (5, 1)
    assert (s["mean_abs_diff"], s["max_abs_diff"]) == (0.03, 0.06)
    assert s["same_level"] == 0.5
    assert s["same_skill_kind"] == 0.75  # pair 1 agrees on its skill, pair 2 on one of two
    assert s["with_unjudged"] == 0.25
    assert s["cost_full_labelling_usd"] == 1.0
