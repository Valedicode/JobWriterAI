"""Run the existing JobWriterAI scoring pipeline (the teacher) over HF (resume, job) text pairs.

The HF `label` column is deliberately NOT read or stored: train-split labels are never used,
test-split labels are reserved for external evaluation.

Run from backend/:
  python -m ml.labeling.generate_teacher_labels --limit 200 --out pilot.jsonl      # pilot
  python -m ml.labeling.generate_teacher_labels --summary-only --out pilot.jsonl   # re-summarize
Resumable: rows already present in the output file are skipped.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from langchain_community.callbacks import get_openai_callback

from app.agents.job_agent import retrieve_text
from app.agents.scoring_agent import _calculate_compatibility_score_v2_internal
from app.models.schemas import JobRequirements
from ml.config import HF_CACHE, HF_DATASET, SEED, TEACHER_LABELS
from ml.labeling.resume_text_extractor import extract_resume_info_from_text

# Default rationale scoring_agent assigns when neither cascade nor LLM resolved a skill;
# seeing it after the LLM step means the batched LLM call failed silently (it swallows errors).
UNJUDGED = "No direct, family, or semantic match found in candidate skills."


def label_row(row_id, resume_text, job_text):
    rec = {"row_id": row_id, "resume_text": resume_text, "job_description_text": job_text}
    t0 = time.perf_counter()
    try:
        with get_openai_callback() as cb:
            job = JobRequirements.model_validate_json(retrieve_text.invoke({"info": job_text}))
            cv = extract_resume_info_from_text(resume_text)
            rec["cv_data"], rec["job_data"] = cv.model_dump(), job.model_dump()
            if not cv.skills and not cv.experience:
                rec["status"] = "degenerate_cv"
            else:
                report = _calculate_compatibility_score_v2_internal(rec["cv_data"], rec["job_data"])
                rec["report"] = report.model_dump()
                gap = report.gap_analysis
                all_matches = gap.matched_skills + gap.transferable_skills + gap.missing_skills
                flags = []
                if any(m.rationale == UNJUDGED for m in all_matches):
                    flags.append("llm_transferability_unjudged")
                if any(d.rationale.startswith("Domain embedding failed") for d in report.dimensions):
                    flags.append("domain_embedding_failed")
                rec["flags"] = flags
                rec["status"] = "ok"
        rec["llm_cost_usd"], rec["llm_tokens"] = cb.total_cost, cb.total_tokens
    except Exception as e:
        rec["status"], rec["error"] = "error", f"{type(e).__name__}: {e}"
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    return rec


def summarize(path, total_rows):
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    ok = [r for r in recs if r["status"] == "ok"]
    clean = [r for r in ok if not r["flags"]]
    cost = [r.get("llm_cost_usd", 0.0) for r in recs]
    lat = [r["latency_s"] for r in recs]
    scores = [r["report"]["aggregate_score"] for r in clean]
    s = {
        "rows": len(recs),
        "status": {k: sum(r["status"] == k for r in recs) for k in ("ok", "degenerate_cv", "error")},
        "flagged_ok_rows": {f: sum(f in r["flags"] for r in ok)
                            for f in ("llm_transferability_unjudged", "domain_embedding_failed")},
        "clean_rows": len(clean),
        "llm_cost_usd": {"total": round(sum(cost), 4), "per_row": round(float(np.mean(cost)), 5),
                         f"extrapolated_{total_rows}_rows": round(float(np.mean(cost)) * total_rows, 2)},
        "latency_s_per_row": {"p50": round(float(np.percentile(lat, 50)), 1),
                              "p95": round(float(np.percentile(lat, 95)), 1)},
        "level_distribution": {lv: sum(r["report"]["level"] == lv for r in clean)
                               for lv in ("low", "medium", "high", "excellent")},
        "aggregate_score": {"mean": round(float(np.mean(scores)), 3), "std": round(float(np.std(scores)), 3),
                            "min": round(min(scores), 3), "max": round(max(scores), 3)} if scores else None,
        "note": "cost excludes OpenAI embedding calls (text-embedding-3-small, negligible)",
        "errors": [r["error"] for r in recs if r["status"] == "error"][:5],
    }
    print(json.dumps(s, indent=2))
    path.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="pilot.jsonl")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE))[args.split]
    out = TEACHER_LABELS / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        done = {json.loads(l)["row_id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
        # Seeded random sample so the pilot isn't biased by dataset row order.
        ids = np.random.default_rng(SEED).permutation(len(ds))[: args.limit]
        todo = [int(i) for i in ids if int(i) not in done]
        print(f"{len(todo)} rows to label ({len(done)} already done) -> {out}")
        with ThreadPoolExecutor(args.workers) as pool, out.open("a", encoding="utf-8") as f:
            futs = [pool.submit(label_row, i, ds[i]["resume_text"], ds[i]["job_description_text"]) for i in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{n}/{len(todo)}] row {rec['row_id']} {rec['status']} {rec['latency_s']}s")
    summarize(out, len(ds))


if __name__ == "__main__":
    main()
