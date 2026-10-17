"""Download the HF resume/job fit dataset and write a due-diligence report.

Run from backend/:  python -m ml.dataprep.prepare_hf_eval_dataset
Output: data/ml/hf_eval_cache/dataset_report.json
"""
import json
from collections import Counter

import numpy as np
from datasets import load_dataset

from ml.config import HF_CACHE, HF_DATASET

TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def percentiles(xs):
    return {f"p{p}": int(np.percentile(xs, p)) for p in (50, 90, 95, 99)} | {"max": int(max(xs))}


def main():
    ds = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        count = lambda texts: [len(ids) for ids in tok(texts)["input_ids"]]
        unit = f"tokens ({TOKENIZER})"
    except ImportError:
        count = lambda texts: [len(t.split()) for t in texts]
        unit = "whitespace words"

    report = {"dataset": HF_DATASET, "length_unit": unit, "splits": {}}
    for name, split in ds.items():
        resumes, jobs = list(split["resume_text"]), list(split["job_description_text"])
        r_len, j_len = count(resumes), count(jobs)
        report["splits"][name] = {
            "rows": len(split),
            "labels": dict(Counter(split["label"])),
            "empty_resume": sum(not t.strip() for t in resumes),
            "empty_job": sum(not t.strip() for t in jobs),
            "unique_resumes": len(set(resumes)),
            "unique_jobs": len(set(jobs)),
            "duplicate_pairs": len(split) - len(set(zip(resumes, jobs))),
            "resume_len": percentiles(r_len),
            "job_len": percentiles(j_len),
            "pair_len": percentiles([a + b for a, b in zip(r_len, j_len)]),
        }

    (tr_r, tr_j), (te_r, te_j) = [(list(ds[s]["resume_text"]), list(ds[s]["job_description_text"]))
                                  for s in ("train", "test")]
    report["train_test_overlap"] = {
        "resumes": len(set(tr_r) & set(te_r)),
        "jobs": len(set(tr_j) & set(te_j)),
        "pairs": len(set(zip(tr_r, tr_j)) & set(zip(te_r, te_j))),
    }

    out = HF_CACHE / "dataset_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
