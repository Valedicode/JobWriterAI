"""Metrics, thresholds and the job-clustered bootstrap (NLP_PROJECT_PLAN.md §9, ml/PREREGISTRATION.md).

Kept to numpy, so every number can be checked by hand in the tests.
"""
import numpy as np

from ml.labeling.label_files import spearman

LABELS = ("No Fit", "Potential Fit", "Good Fit")
ORDINAL = {label: k for k, label in enumerate(LABELS)}
BOOTSTRAP_REPLICATES = 2000


def safe_spearman(x, y) -> float:
    """Spearman's rho, or nan when either side is constant (e.g. the majority class)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return spearman(x, y) if x.std() > 0 and y.std() > 0 else float("nan")


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def mae(x, y) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))))


def accuracy(true, pred) -> float:
    return float(np.mean(np.asarray(true) == np.asarray(pred)))


def macro_f1(true, pred, k: int = len(LABELS)) -> float:
    """Mean F1 over all k classes; a class never predicted and never present counts as 0."""
    true, pred = np.asarray(true), np.asarray(pred)
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (true == c))
        denom = np.sum(pred == c) + np.sum(true == c)
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def qwk(true, pred, k: int = len(LABELS)) -> float:
    """Quadratic-weighted Cohen's kappa."""
    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    observed = np.zeros((k, k))
    np.add.at(observed, (true, pred), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    weights = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    denom = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denom) if denom else float("nan")


def classify(scores, thresholds: tuple[float, float]) -> np.ndarray:
    """0 below the first threshold, 1 below the second, 2 from the second on."""
    t1, t2 = thresholds
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < t1, 0, np.where(scores < t2, 1, 2))


def fit_thresholds(scores, labels) -> tuple[float, float]:
    """The two thresholds maximising macro-F1, over a grid of the scores' 1st-99th percentiles
    (pre-registered). Ties go to the lowest pair."""
    grid = np.unique(np.percentile(np.asarray(scores, dtype=float), np.arange(1, 100)))
    best, best_f1 = (float(grid[0]), float(grid[-1])), -1.0
    for i, t1 in enumerate(grid):
        for t2 in grid[i + 1:]:
            f1 = macro_f1(labels, classify(scores, (t1, t2)))
            if f1 > best_f1:
                best, best_f1 = (float(t1), float(t2)), f1
    return best


def cluster_replicates(groups, n: int = BOOTSTRAP_REPLICATES, seed: int = 42):
    """Row indices of n bootstrap samples that draw whole groups (jobs) with replacement."""
    groups = np.asarray(groups)
    rows_of = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.concatenate([rows_of[g] for g in rng.integers(0, len(rows_of), len(rows_of))])


def bootstrap_ci(stat, groups, n: int = BOOTSTRAP_REPLICATES, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI of stat(rows) under the job-clustered bootstrap."""
    reps = np.array([stat(rows) for rows in cluster_replicates(groups, n, seed)])
    if np.isnan(reps).all():  # e.g. rho of a constant score
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def paired_difference(stat_a, stat_b, groups, n: int = BOOTSTRAP_REPLICATES, seed: int = 42) -> dict:
    """stat_a - stat_b on the same resampled jobs: the observed difference, its 95% CI and a
    two-sided bootstrap p-value."""
    everything = np.arange(len(groups))
    diffs = np.array([stat_a(rows) - stat_b(rows) for rows in cluster_replicates(groups, n, seed)])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {"difference": float(stat_a(everything) - stat_b(everything)),
            "ci": [float(lo), float(hi)], "p": float(p)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
