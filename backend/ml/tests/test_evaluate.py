"""Unparsed outputs get the system's val mean, and the baselines' features see the pair's overlap.

Run from backend/:  python -m pytest ml/tests
"""
import pytest

from ml.evaluation.baselines import resume_prior
from ml.evaluation.evaluate import aligned

PREDICTION = {"scale": "teacher", "rows": [
    {"row_id": 1, "split": "val", "score": 0.2}, {"row_id": 2, "split": "val", "score": 0.6},
    {"row_id": 3, "split": "val", "score": None}, {"row_id": 9, "split": "test", "score": 0.9}]}


def test_an_unparsed_or_missing_row_gets_the_val_mean():
    scores, classes, coverage = aligned(PREDICTION, "val", [1, 2, 3])
    assert scores.tolist() == pytest.approx([0.2, 0.6, 0.4]) and classes is None
    assert coverage == pytest.approx(2 / 3)
    scores, _, coverage = aligned(PREDICTION, "test", [9, 10])
    assert scores.tolist() == pytest.approx([0.9, 0.4]) and coverage == 0.5


def test_predicted_classes_are_kept_when_every_row_has_one():
    prediction = {"scale": "other", "rows": [{"row_id": i, "split": "val", "score": 0.0, "label": 0} for i in (1, 2)]}
    assert aligned(prediction, "val", [1, 2])[1].tolist() == [0, 0]


def test_the_resume_prior_falls_back_to_the_overall_mean():
    prior = resume_prior(["a", "a", "b"], [0, 2, 2])
    assert prior("a") == 1.0 and prior("b") == 2.0 and prior("unseen") == pytest.approx(4 / 3)
