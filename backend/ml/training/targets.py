"""The students' targets (NLP_PROJECT_PLAN.md §7.1), and their export for training.

`targets.jsonl` holds, per clean `train` and `val` row: the row id, its split, Student-T's target
(the teacher's scores) and the dataset label (Student-D's target). It has no resume or job text, so
it can be uploaded to Kaggle; the training script reads the texts from the HF dataset by row id.

Run from backend/, after the labels and ml.dataprep.splits:
  python -m ml.training.targets
Output: data/ml/students/targets.jsonl
"""
import json

from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, HF_REVISION, TEACHER_LABELS
from ml.dataprep.splits import load_split
from ml.labeling.label_files import latest, read_jsonl

DIMENSIONS = ("hard_skills", "experience", "seniority", "domain", "ats_keywords")
LABELS = ("No Fit", "Potential Fit", "Good Fit")
TARGETS = ARTIFACTS / "students" / "targets.jsonl"


def teacher_target(report: dict) -> str:
    """Dimensions first, so the aggregate is generated after them, as the teacher computes it."""
    dims = {d["name"]: round(d["score"], 2) for d in report["dimensions"]}
    return json.dumps({"dimensions": {k: dims[k] for k in DIMENSIONS},
                       "aggregate_score": round(report["aggregate_score"], 2), "level": report["level"]})


def dataset_target(label: str) -> str:
    return json.dumps({"label": label})


def build(split: dict[str, list[int]], labelled: dict[int, dict], labels: list[str]) -> list[dict]:
    """One record per clean teacher-labelled row of `train` and `val`, in split order."""
    out = []
    for name in ("train", "val"):
        for i in split[name]:
            r = labelled.get(i)
            if r and r["status"] == "ok" and not r["flags"]:
                out.append({"row_id": i, "split": name, "teacher": teacher_target(r["report"]), "label": labels[i]})
    return out


def main():
    from datasets import load_dataset
    labels = load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))["train"]["label"]
    split = load_split()
    records = build(split, latest(read_jsonl(TEACHER_LABELS / "train.jsonl")), labels)
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    TARGETS.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    counts = {name: sum(r["split"] == name for r in records) for name in ("train", "val")}
    dropped = {name: len(split[name]) - counts[name] for name in counts}
    print(f"{counts} rows (not clean, left out: {dropped}) -> {TARGETS}")


if __name__ == "__main__":
    main()
