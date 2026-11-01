"""The job-grouped train/val split of the HF train split (NLP_PROJECT_PLAN.md §6).

Row ids are the HF train split's row indices. A repeat of an earlier (resume, job) pair is dropped,
as in labelling. About 12% of the distinct jobs, with all their rows, form `val`; the rest is
`train`. The HF test split is not touched.

Jobs are ordered by the sha256 of their text before the seeded shuffle, so the split depends only
on the texts and the seed, not on the row order.

Run from backend/:  python -m ml.dataprep.splits
Output: data/ml/splits/splits.json (row ids per split, the val jobs' sha256, counts)
"""
import hashlib
import json
from collections import Counter

import numpy as np

from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, HF_REVISION, SEED

VAL_JOB_SHARE = 0.12
SPLITS = ARTIFACTS / "splits" / "splits.json"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_rows(resumes: list[str], jobs: list[str]) -> list[int]:
    """Row ids without repeats of an earlier (resume, job) pair."""
    seen, rows = set(), []
    for i, pair in enumerate(zip(resumes, jobs)):
        if pair not in seen:
            seen.add(pair)
            rows.append(i)
    return rows


def job_split(resumes: list[str], jobs: list[str], share: float = VAL_JOB_SHARE,
              seed: int = SEED) -> dict[str, list[int]]:
    """Row ids of `train` and `val`, with every row of a job on the same side."""
    rows = unique_rows(resumes, jobs)
    distinct = sorted({sha256(jobs[i]) for i in rows})
    order = np.random.default_rng(seed).permutation(len(distinct))
    val_jobs = {distinct[k] for k in order[:round(share * len(distinct))]}
    val = [i for i in rows if sha256(jobs[i]) in val_jobs]
    taken = set(val)
    return {"train": [i for i in rows if i not in taken], "val": val}


def load_split() -> dict[str, list[int]]:
    """The saved split (run this module once to write it)."""
    return json.loads(SPLITS.read_text(encoding="utf-8"))["rows"]


def main():
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))["train"]
    resumes, jobs, labels = ds["resume_text"], ds["job_description_text"], ds["label"]
    split = job_split(resumes, jobs)
    summary = {name: {"rows": len(ids), "jobs": len({sha256(jobs[i]) for i in ids}),
                      "resumes": len({sha256(resumes[i]) for i in ids}),
                      "labels": dict(Counter(labels[i] for i in ids))} for name, ids in split.items()}
    train_resumes = {sha256(resumes[i]) for i in split["train"]}
    summary["val"]["rows_with_resume_in_train"] = sum(sha256(resumes[i]) in train_resumes for i in split["val"])
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(json.dumps({
        "dataset": {"id": HF_DATASET, "revision": HF_REVISION, "split": "train"},
        "seed": SEED, "val_job_share": VAL_JOB_SHARE, "summary": summary,
        "val_jobs_sha256": sorted({sha256(jobs[i]) for i in split["val"]}), "rows": split}), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {SPLITS}")


if __name__ == "__main__":
    main()
