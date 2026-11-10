"""Prediction files and the reference rows they are scored against.

Every system writes one file, `predictions/<system>.json`, with a continuous score per `val` and
test row (and, for a system that only predicts a class, that class):

    {"system": "tfidf-ridge", "scale": "teacher", "rows": [{"row_id": 7, "split": "val", "score": 0.41}, ...]}

`scale` is "teacher" when the score is on the teacher's 0-1 aggregate scale (so its MAE against the
teacher means something), else "other". A score of null marks an output that couldn't be parsed.
"""
import hashlib
import json

from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, HF_REVISION, TEACHER_LABELS
from ml.dataprep.splits import load_split
from ml.evaluation.metrics import ORDINAL
from ml.labeling.label_files import latest, read_jsonl

PREDICTIONS = ARTIFACTS / "predictions"
SPLITS = ("val", "test")


def write(system: str, scale: str, rows: list[dict]) -> None:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS / f"{system}.json"
    path.write_text(json.dumps({"system": system, "scale": scale, "rows": rows}), encoding="utf-8")
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    print(f"{system}: {counts} -> {path}")


def read_all() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PREDICTIONS.glob("*.json"))}


def sha256_of(system: str) -> str:
    return hashlib.sha256((PREDICTIONS / f"{system}.json").read_bytes()).hexdigest()


def hf_split(name: str):
    from datasets import load_dataset
    return load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))[name]


def row_ids(split: str) -> list[int]:
    """`val` rows index the HF train split; test rows index the HF test split."""
    return load_split()["val"] if split == "val" else sorted(latest(read_jsonl(TEACHER_LABELS / "test.jsonl")))


def reference(split: str, with_labels: bool) -> dict[str, list]:
    """Per row of `split`: its id, its job (for the bootstrap), the teacher's aggregate and, only
    when asked, the dataset label as 0/1/2. The test labels are read only by the final evaluation."""
    labelled = latest(read_jsonl(TEACHER_LABELS / ("train.jsonl" if split == "val" else "test.jsonl")))
    ids = row_ids(split)
    ref = {"row_id": ids,
           "job": [labelled[i]["job_sha256"] for i in ids],
           "teacher": [labelled[i]["report"]["aggregate_score"] for i in ids]}
    if with_labels:
        labels = hf_split("train" if split == "val" else "test")["label"]
        ref["label"] = [ORDINAL[labels[i]] for i in ids]
    return ref
