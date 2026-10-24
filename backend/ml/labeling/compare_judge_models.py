"""
Compare candidate judge models for the teacher's score, before freezing the teacher.

The score has one LLM call of its own: the batched skill-transferability verdicts
(chat_model("judge")). Given the extracted CV and job, everything else in it is deterministic,
so the judges are compared on the same extractions:

1. --extract: N seeded train pairs; the job and the CV are extracted once with the extract
   model the settings route to (set LLM_MODEL_EXTRACT), and saved.
2. --judges a,b: every pair is scored --runs times with each judge.

Per judge: failed scores, scores with unjudged skills (the batched call failed silently), how
far two runs of the same pair are apart (mean and max |difference| of the aggregate score, same
level, skills given the same kind), the mean score, latency, and cost from the token counts.

Run from backend/ (needs OPENROUTER_API_KEY):
  LLM_MODEL_EXTRACT=mistralai/mistral-large-4-0 python -m ml.labeling.compare_judge_models --extract --pairs 20
  python -m ml.labeling.compare_judge_models --judges deepseek/deepseek-v4.1-flash-1017,minimax/minimax-m3
Results: data/ml/judge_comparison/ (extractions.jsonl; <timestamp>/results.jsonl, summary.json, summary.md)
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import mean

import numpy as np
from datasets import load_dataset
from langchain_community.callbacks import get_openai_callback

from app.agents.job_agent import retrieve_text
from app.agents.scoring_agent import _calculate_compatibility_score_v2_internal
from app.config import get_settings
from app.models.schemas import JobRequirements
from app.openai_llm import model_for
from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, SEED
from ml.labeling.compare_extraction_models import load_prices
from ml.labeling.teacher import UNJUDGED
from ml.labeling.resume_text_extractor import extract_resume_info_from_text

OUT = ARTIFACTS / "judge_comparison"
EXTRACTIONS = OUT / "extractions.jsonl"


def extract_pairs(n: int, workers: int) -> None:
    ds = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE))["train"]
    ids = [int(i) for i in np.random.default_rng(SEED).permutation(len(ds))[:n]]
    extract_model = model_for("extract")

    def one(row_id: int) -> dict:
        row = ds[row_id]
        job = JobRequirements.model_validate_json(retrieve_text.invoke({"info": row["job_description_text"]}))
        cv = extract_resume_info_from_text(row["resume_text"])
        return {"row_id": row_id, "extract_model": extract_model, "cv_data": cv.model_dump(),
                "job_data": job.model_dump()}

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"extracting {len(ids)} pairs with {extract_model} -> {EXTRACTIONS}")
    with ThreadPoolExecutor(workers) as pool, EXTRACTIONS.open("w", encoding="utf-8") as f:
        futures = {pool.submit(one, i): i for i in ids}
        for n_done, fut in enumerate(as_completed(futures), 1):
            try:
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                print(f"[{n_done}/{len(ids)}] row {futures[fut]} ok")
            except Exception as e:  # noqa: BLE001 - a failed extraction just leaves the pair out
                print(f"[{n_done}/{len(ids)}] row {futures[fut]} failed: {type(e).__name__}: {str(e)[:200]}")


def _kinds(report) -> dict[str, str]:
    gap = report.gap_analysis
    buckets = (gap.matched_skills, gap.transferable_skills, gap.learnable_skills, gap.missing_skills)
    return {m.required_skill: m.kind for bucket in buckets for m in bucket}


def score_once(judge: str, pair: dict, run: int, prices: dict) -> dict:
    rec = {"judge": judge, "row_id": pair["row_id"], "run": run}
    t0 = time.perf_counter()
    try:
        with get_openai_callback() as cb:
            report = _calculate_compatibility_score_v2_internal(pair["cv_data"], pair["job_data"])
        gap = report.gap_analysis
        matches = gap.matched_skills + gap.transferable_skills + gap.learnable_skills + gap.missing_skills
        price_in, price_out = prices.get(judge, (0.0, 0.0))
        rec.update(status="ok", score=report.aggregate_score, level=report.level, kinds=_kinds(report),
                   unjudged=sum(m.rationale == UNJUDGED for m in matches),
                   cost_usd=cb.prompt_tokens * price_in + cb.completion_tokens * price_out)
    except Exception as e:  # noqa: BLE001 - a failed score is a result for the comparison
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    rec["latency_s"] = round(time.perf_counter() - t0, 1)
    return rec


def summarize(records: list[dict], judges: list[str], full_rows: int) -> dict:
    summary = {}
    for judge in judges:
        recs = [r for r in records if r["judge"] == judge]
        ok = [r for r in recs if r["status"] == "ok"]
        by_pair: dict[int, list[dict]] = {}
        for r in ok:
            by_pair.setdefault(r["row_id"], []).append(r)
        pairs = [sorted(rs, key=lambda r: r["run"])[:2] for rs in by_pair.values() if len(rs) >= 2]
        diffs = [abs(a["score"] - b["score"]) for a, b in pairs]
        same_kind = []
        for a, b in pairs:
            skills = set(a["kinds"]) | set(b["kinds"])
            same_kind.append(mean(a["kinds"].get(s) == b["kinds"].get(s) for s in skills) if skills else 1.0)
        summary[judge] = {
            "scores": len(recs),
            "failed": len(recs) - len(ok),
            "errors": sorted({r["error"][:120] for r in recs if r["status"] == "error"})[:3],
            "with_unjudged": round(mean(r["unjudged"] > 0 for r in ok), 2) if ok else None,
            "mean_abs_diff": round(mean(diffs), 4) if diffs else None,
            "max_abs_diff": round(max(diffs), 4) if diffs else None,
            "same_level": round(mean(a["level"] == b["level"] for a, b in pairs), 2) if pairs else None,
            "same_skill_kind": round(mean(same_kind), 3) if same_kind else None,
            "mean_score": round(mean(r["score"] for r in ok), 3) if ok else None,
            "latency_s_median": round(float(np.median([r["latency_s"] for r in ok])), 1) if ok else None,
            "cost_per_score_usd": round(mean(r["cost_usd"] for r in ok), 5) if ok else None,
            "cost_full_labelling_usd": round(mean(r["cost_usd"] for r in ok) * full_rows, 2) if ok else None,
        }
    return summary


def to_markdown(summary: dict) -> str:
    cols = ["failed", "with_unjudged", "mean_abs_diff", "max_abs_diff", "same_level", "same_skill_kind",
            "mean_score", "latency_s_median", "cost_per_score_usd", "cost_full_labelling_usd"]
    lines = ["| judge | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for judge, s in summary.items():
        lines.append(f"| {judge} | " + " | ".join(str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


def compare_judges(judges: list[str], runs: int, workers: int, catalog: str | None, full_rows: int) -> None:
    prices = load_prices(catalog)
    unknown = [j for j in judges if j not in prices]
    if unknown:
        raise SystemExit(f"not in the OpenRouter catalogue: {unknown}")
    pairs = [json.loads(line) for line in EXTRACTIONS.open(encoding="utf-8")]
    out_dir = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(pairs)} pairs x {runs} runs x {len(judges)} judges -> {out_dir}")

    records = []
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        # One judge at a time: the judge is a setting, read when the score calls the model.
        for judge in judges:
            os.environ["LLM_MODEL_JUDGE"] = judge
            get_settings.cache_clear()
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(score_once, judge, p, run, prices)
                           for p in pairs for run in range(1, runs + 1)]
                for n_done, fut in enumerate(as_completed(futures), 1):
                    rec = fut.result()
                    records.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{judge} {n_done}/{len(futures)}] row {rec['row_id']} run {rec['run']}: "
                          f"{rec['status']} {rec.get('score', '')} {rec['latency_s']}s")

    summary = summarize(records, judges, full_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary) + "\n", encoding="utf-8")
    print(to_markdown(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="extract the pairs (once, before comparing judges)")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--judges", default="", help="comma-separated OpenRouter model ids")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--catalog", help="a saved copy of the OpenRouter catalogue (else it is fetched)")
    # Every train and test row is labelled (the pilot's extrapolation base).
    ap.add_argument("--full-rows", type=int, default=6241)
    args = ap.parse_args()

    if args.extract:
        extract_pairs(args.pairs, args.workers)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if judges:
        compare_judges(judges, args.runs, args.workers, args.catalog, args.full_rows)


if __name__ == "__main__":
    main()
