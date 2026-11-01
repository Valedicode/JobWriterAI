"""The students' targets put the dimensions first and keep only clean train and val rows.

Run from backend/:  python -m pytest ml/tests
"""
import json

from ml.training.targets import build, dataset_target, teacher_target

REPORT = {"aggregate_score": 0.38149, "level": "low",
          "dimensions": [{"name": "domain", "score": 0.41}, {"name": "hard_skills", "score": 0.3249},
                         {"name": "experience", "score": 0.5}, {"name": "seniority", "score": 0.6},
                         {"name": "ats_keywords", "score": 0.178}, {"name": "eligibility", "score": 1.0}]}


def test_the_teacher_target_lists_the_dimensions_before_the_aggregate():
    text = teacher_target(REPORT)
    assert text.index("dimensions") < text.index("aggregate_score") < text.index("level")
    assert json.loads(text) == {"dimensions": {"hard_skills": 0.32, "experience": 0.5, "seniority": 0.6,
                                               "domain": 0.41, "ats_keywords": 0.18},
                                "aggregate_score": 0.38, "level": "low"}


def test_the_dataset_target_is_the_label():
    assert json.loads(dataset_target("Good Fit")) == {"label": "Good Fit"}


def test_only_clean_rows_of_train_and_val_are_kept():
    ok = {"status": "ok", "flags": [], "report": REPORT}
    labelled = {0: ok, 1: {**ok, "flags": ["domain_embedding_failed"]}, 2: {"status": "error", "flags": []}, 3: ok}
    records = build({"train": [0, 1, 2], "val": [3, 4]}, labelled, ["No Fit", "Good Fit", "No Fit", "Potential Fit", "No Fit"])
    assert [(r["row_id"], r["split"], r["label"]) for r in records] == [(0, "train", "No Fit"), (3, "val", "Potential Fit")]
