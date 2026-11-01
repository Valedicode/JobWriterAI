"""The val split keeps each job on one side, and the audit sample spreads over val's jobs.

Run from backend/:  python -m pytest ml/tests
"""
from collections import Counter

from ml.dataprep.audit_sample import draw
from ml.dataprep.splits import job_split, unique_rows


def _data():
    jobs = [f"job{j}" for j in range(20) for _ in range(5)]
    resumes = [f"cv{k % 13}" for k in range(100)]
    return resumes, jobs


def test_a_repeated_pair_is_dropped():
    assert unique_rows(["a", "b", "a"], ["x", "y", "x"]) == [0, 1]


def test_each_job_falls_on_one_side():
    resumes, jobs = _data()
    split = job_split(resumes, jobs, share=0.2)
    val_jobs = {jobs[i] for i in split["val"]}
    assert len(val_jobs) == 4 and not val_jobs & {jobs[i] for i in split["train"]}
    assert sorted(split["train"] + split["val"]) == list(range(100))


def test_the_split_ignores_row_order():
    resumes, jobs = _data()
    order = list(reversed(range(100)))
    split = job_split([resumes[i] for i in order], [jobs[i] for i in order], share=0.2)
    assert {jobs[order[i]] for i in split["val"]} == {jobs[i] for i in job_split(resumes, jobs, share=0.2)["val"]}


def test_the_audit_takes_from_each_job_in_turn():
    resumes, jobs = _data()
    drawn = draw(list(range(100)), jobs, size=30)
    assert len(set(drawn)) == 30
    assert max(Counter(jobs[i] for i in drawn).values()) <= 2  # 30 rows over 20 jobs
    assert draw(list(range(100)), jobs, size=30) == drawn
