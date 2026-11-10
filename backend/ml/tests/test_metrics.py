"""The metrics against hand-computed cases.

Run from backend/:  python -m pytest ml/tests
"""
import math

import numpy as np
import pytest

from ml.evaluation.metrics import (classify, cluster_replicates, fit_thresholds, holm, macro_f1, paired_difference,
                                   qwk, safe_spearman)


def test_macro_f1_averages_every_class():
    # class 0: tp 1, 2 predicted, 1 true -> 2/3; class 1: tp 1, 1 predicted, 2 true -> 2/3; class 2: never -> 0
    assert macro_f1([0, 1, 1], [0, 1, 0]) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_qwk_is_one_when_perfect_and_penalises_far_misses_more():
    assert qwk([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    near = qwk([0, 1, 2, 2], [0, 1, 1, 2])
    far = qwk([0, 1, 2, 2], [0, 1, 0, 2])
    assert far < near < 1.0


def test_qwk_by_hand():
    # k=2, so a miss weighs 1. Observed misses: one (1,0). Expected = outer(row sums [1,2],
    # column sums [2,1]) / 3, whose off-diagonal cells sum to (1 + 4) / 3. kappa = 1 - 1 / (5/3).
    true, pred = [0, 1, 1], [0, 0, 1]
    expected_disagreement = (1 * 1 + 2 * 2) / 3
    assert qwk(true, pred, k=2) == pytest.approx(1 - 1 / expected_disagreement)


def test_classify_uses_half_open_bands():
    assert classify([0.1, 0.3, 0.5, 0.7], (0.3, 0.7)).tolist() == [0, 1, 1, 2]


def test_fit_thresholds_recovers_a_clean_separation():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 50)
    scores = labels + rng.uniform(0, 0.5, 150)
    assert macro_f1(labels, classify(scores, fit_thresholds(scores, labels))) == pytest.approx(1.0)


def test_spearman_of_a_constant_is_nan():
    assert math.isnan(safe_spearman([1, 1, 1], [0, 1, 2]))
    assert safe_spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_the_bootstrap_draws_whole_jobs():
    groups = ["a", "a", "b", "b", "b", "c"]
    for rows in cluster_replicates(groups, n=50, seed=1):
        drawn = [groups[i] for i in rows]
        for g in set(drawn):
            assert drawn.count(g) % groups.count(g) == 0


def test_a_paired_difference_of_a_system_with_itself_is_zero():
    groups = np.repeat(np.arange(10), 5)
    x = np.random.default_rng(0).normal(size=50)
    stat = lambda rows: float(np.mean(x[rows]))
    result = paired_difference(stat, stat, groups, n=200)
    assert result["difference"] == 0 and result["ci"] == [0.0, 0.0] and result["p"] == 1.0


def test_holm_by_hand():
    # sorted 0.01, 0.02, 0.04 -> 3*0.01, 2*0.02, max(0.04, 1*0.04); returned in input order
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
