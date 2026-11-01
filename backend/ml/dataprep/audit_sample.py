"""Draw the manual audit set from `val` (NLP_PROJECT_PLAN.md §6, ml/AUDIT_RUBRIC.md).

80 pairs, spread over val's jobs: each job's rows are shuffled, then rows are taken from the jobs
in turn until 80 are drawn. The first 30 are the overlap set for a second annotator.

The files are blind: they hold the texts and empty label columns, never the dataset's label or the
teacher's score. They contain resume text, so they stay in the git-ignored data/ml/audit/.

Run from backend/, after ml.dataprep.splits:  python -m ml.dataprep.audit_sample
Output: data/ml/audit/audit_sheet.csv (fill in) and audit_pairs.md (the texts to read)
"""
import csv

import numpy as np

from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, HF_REVISION, SEED
from ml.dataprep.splits import load_split, sha256

AUDIT_SIZE = 80
OVERLAP = 30
AUDIT = ARTIFACTS / "audit"
COLUMNS = ("audit_id", "row_id", "overlap", "label", "confidence", "notes")


def draw(val_rows: list[int], jobs: list[str], size: int = AUDIT_SIZE, seed: int = SEED) -> list[int]:
    """Row ids, taken from each job in turn (jobs and their rows in a seeded order)."""
    rng = np.random.default_rng(seed)
    by_job: dict[str, list[int]] = {}
    for i in val_rows:
        by_job.setdefault(sha256(jobs[i]), []).append(i)
    queues = [list(rng.permutation(by_job[j])) for j in sorted(by_job)]
    queues = [queues[k] for k in rng.permutation(len(queues))]
    drawn = []
    while len(drawn) < size and any(queues):
        for q in queues:
            if q and len(drawn) < size:
                drawn.append(int(q.pop(0)))
    return drawn


def main():
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))["train"]
    resumes, jobs = ds["resume_text"], ds["job_description_text"]
    rows = draw(load_split()["val"], jobs)
    AUDIT.mkdir(parents=True, exist_ok=True)
    sheet = AUDIT / "audit_sheet.csv"
    if sheet.exists():
        raise SystemExit(f"{sheet} exists; it may hold labels already. Move it away to draw again.")
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for n, i in enumerate(rows, 1):
            w.writerow([n, i, "yes" if n <= OVERLAP else "no", "", "", ""])
    with (AUDIT / "audit_pairs.md").open("w", encoding="utf-8") as f:
        f.write("# Audit pairs\n\nLabel each pair in audit_sheet.csv following ml/AUDIT_RUBRIC.md.\n")
        for n, i in enumerate(rows, 1):
            f.write(f"\n---\n\n## Pair {n} (row {i}{', overlap' if n <= OVERLAP else ''})\n\n"
                    f"### Job\n\n{jobs[i].strip()}\n\n### Resume\n\n{resumes[i].strip()}\n")
    print(f"{len(rows)} pairs from {len({sha256(jobs[i]) for i in rows})} jobs -> {AUDIT}")


if __name__ == "__main__":
    main()
