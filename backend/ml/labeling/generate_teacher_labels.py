"""Label HF (resume, job) pairs with the teacher: the app's extraction and scoring (ml.labeling.teacher).

A run has two steps. First every distinct resume and job text is extracted once, through the
extraction cache. Then each row is scored from its two extractions. The HF `label` column is never
read while labelling; only the summary of a train run compares train labels with the scores.
Test labels are never read here.

Run from backend/ (ml/README.md has the frozen teacher's settings):
  python -m ml.labeling.generate_teacher_labels --limit 200 --out repilot.jsonl          # re-pilot
  python -m ml.labeling.generate_teacher_labels --limit 0 --workers 16 --out train.jsonl
  python -m ml.labeling.generate_teacher_labels --summary-only --out repilot.jsonl
Resumable: rows already labelled (ok or degenerate) are skipped and failed rows are tried again.
A fatal API error (bad key, no credit, region refused) or too many failures in a row stops the run;
everything finished so far is kept, and rerunning the same command continues it.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from datasets import load_dataset
from langchain_community.callbacks import get_openai_callback

from ml.config import HF_CACHE, HF_DATASET, HF_REVISION, SEED, TEACHER_CACHE, TEACHER_LABELS
from ml.labeling import teacher, teacher_meta
from ml.labeling.caches import EmbeddingCache, ExtractionCache, Store, sha256
from ml.labeling.label_files import LEVELS, latest, read_jsonl, spearman

ATTEMPTS = 4
BACKOFF_S = 5.0
MAX_ERRORS_IN_A_ROW = 25
DONE = {"ok", "degenerate_cv"}
LABEL_ORDER = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


class StopRun(RuntimeError):
    """Too many failures in a row: the problem is not with single rows."""


def select_rows(resumes: list[str], jobs: list[str], limit: int, seed: int = SEED) -> list[int]:
    """The row ids to label: the first ``limit`` of a seeded permutation (the v0 pilot's rows), or
    every row when ``limit`` is 0. A repeat of an earlier (resume, job) pair is left out."""
    first: dict[tuple[str, str], int] = {}
    for i, pair in enumerate(zip(resumes, jobs)):
        first.setdefault(pair, i)
    ids = np.random.default_rng(seed).permutation(len(resumes))[:limit] if limit else range(len(resumes))
    return [int(i) for i in ids if first[(resumes[i], jobs[i])] == i]


def spend(tokens: dict, price: dict | None) -> float | None:
    if not price:
        return None
    return tokens["prompt"] * price["prompt"] + tokens["completion"] * price["completion"]


def _wait(attempt: int, backoff_s: float) -> None:
    time.sleep(backoff_s * 2 ** (attempt - 1) * random.uniform(0.8, 1.2))


def with_retries(fn, attempts: int = ATTEMPTS, backoff_s: float = BACKOFF_S):
    """``(fn(), attempts used)``, retrying failures with exponential backoff. A fatal API error is
    raised at once as FatalAPIError."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt
        except teacher.FatalAPIError:
            raise
        except Exception as e:
            if teacher.is_fatal(e):
                raise teacher.FatalAPIError(f"{type(e).__name__}: {e}") from e
            if attempt == attempts:
                raise
            _wait(attempt, backoff_s)


def extract_text(kind: str, text: str, cache: ExtractionCache, price: dict | None,
                 attempts: int = ATTEMPTS, backoff_s: float = BACKOFF_S) -> dict:
    """Extract one text into the cache. The record keeps the result, for the extraction log."""
    rec = {"kind": kind, "sha256": sha256(text)}
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        try:
            rec["value"], rec["attempts"] = with_retries(lambda: teacher.extract(kind, text), attempts, backoff_s)
            rec["status"] = "ok"
            cache.put(kind, text, rec["value"])
        except teacher.FatalAPIError:
            raise
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}", attempts=attempts)
    rec["tokens"] = {"prompt": cb.prompt_tokens, "completion": cb.completion_tokens}
    rec["cost_usd"] = spend(rec["tokens"], price)
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    return rec


def score_row(row_id: int, resume_text: str, job_text: str, cache: ExtractionCache, price: dict | None,
              attempts: int = ATTEMPTS, backoff_s: float = BACKOFF_S) -> dict:
    """Score one row from its cached extractions. A score the scorer reached despite a failed API
    call is retried; if every attempt fails the row is an error, so the next run tries it again."""
    rec = {"row_id": row_id, "resume_sha256": sha256(resume_text), "job_sha256": sha256(job_text)}
    cv, job = cache.get("resume", resume_text), cache.get("job", job_text)
    if cv is None or job is None:
        return {**rec, "status": "error", "error": "extraction failed (see the .extractions.jsonl log)",
                "attempts": 0, "latency_s": 0.0}
    if teacher.is_degenerate(cv["cv"]):
        return {**rec, "status": "degenerate_cv", "attempts": 0, "latency_s": 0.0}
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        for attempt in range(1, attempts + 1):
            try:
                report, flags, errors = teacher.score(cv["cv"], job["job"])
            except Exception as e:
                report, flags, errors = None, [], [e]
            fatal = [e for e in errors if teacher.is_fatal(e)]
            if fatal:
                raise teacher.FatalAPIError(f"{type(fatal[0]).__name__}: {fatal[0]}") from fatal[0]
            if report is not None and not errors and not flags:
                break
            if attempt < attempts:
                _wait(attempt, backoff_s)
    if report is None or errors:
        rec.update(status="error", error="; ".join(sorted({f"{type(e).__name__}: {e}"[:300] for e in errors})))
    else:
        # Flags left after every attempt (the judge keeps leaving a skill out) are kept and counted;
        # flagged rows are excluded from training.
        rec.update(status="ok", report=report, flags=flags)
    rec["attempts"] = attempt
    rec["tokens"] = {"prompt": cb.prompt_tokens, "completion": cb.completion_tokens}
    rec["cost_usd"] = spend(rec["tokens"], price)
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    return rec


def run_parallel(tasks: list, workers: int, write, describe) -> None:
    """Run the tasks (callables returning a record) on a thread pool, writing each record as it
    finishes. A fatal API error, or MAX_ERRORS_IN_A_ROW failed records in a row, cancels the tasks
    not started yet and is raised; records already written are kept."""
    in_a_row = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(task) for task in tasks]
        try:
            for n, future in enumerate(as_completed(futures), 1):
                rec = future.result()
                write(rec)
                print(f"[{n}/{len(futures)}] {describe(rec)}", flush=True)
                in_a_row = in_a_row + 1 if rec["status"] == "error" else 0
                if in_a_row >= MAX_ERRORS_IN_A_ROW:
                    raise StopRun(f"{in_a_row} failures in a row, the last: {rec.get('error')}")
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _appender(path: Path):
    f = path.open("a", encoding="utf-8")

    def write(rec: dict) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return f, write


def agreement_with_labels(clean: list[dict], labels: list[str]) -> dict:
    """How the teacher's aggregate score tracks the dataset's labels (train only)."""
    scores = [r["report"]["aggregate_score"] for r in clean]
    ordinal = [LABEL_ORDER[labels[r["row_id"]]] for r in clean]
    return {"spearman_aggregate_vs_label": round(spearman(scores, ordinal), 3),
            "mean_aggregate_by_label": {name: round(float(np.mean([s for s, o in zip(scores, ordinal) if o == v])), 3)
                                        for name, v in LABEL_ORDER.items() if v in ordinal}}


def dimension_means(clean: list[dict]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for r in clean:
        for d in r["report"]["dimensions"]:
            scores.setdefault(d["name"], []).append(d["score"])
    return {name: round(float(np.mean(v)), 3) for name, v in scores.items()}


def summarize(out: Path, split: str, labels: list[str] | None = None) -> dict:
    """Totals for a label file: statuses, flags, spend, latency, levels and scores. With ``labels``
    (train only) also the agreement with the dataset's labels."""
    every = read_jsonl(out)
    recs = list(latest(every).values())
    extractions = read_jsonl(out.with_suffix(".extractions.jsonl"))
    ok = [r for r in recs if r["status"] == "ok"]
    clean = [r for r in ok if not r["flags"]]
    scores = [r["report"]["aggregate_score"] for r in clean]
    latency = [r["latency_s"] for r in recs if r["status"] == "ok"]
    s = {
        "split": split,
        "rows": len(recs),
        "status": {k: sum(r["status"] == k for r in recs) for k in ("ok", "degenerate_cv", "error")},
        "flagged_ok_rows": {f: sum(f in r["flags"] for r in ok)
                            for f in ("llm_transferability_unjudged", "domain_embedding_failed")},
        "clean_rows": len(clean),
        "retried_rows": sum(r.get("attempts", 0) > 1 for r in recs),
        # Every attempt costs money, so spend sums all records, including superseded ones.
        "spend_usd": {"scoring": round(sum(r.get("cost_usd") or 0.0 for r in every), 4),
                      "extraction": round(sum(r.get("cost_usd") or 0.0 for r in extractions), 4),
                      "extracted_texts": sum(r["status"] == "ok" for r in extractions)},
        "scoring_latency_s": ({"p50": round(float(np.percentile(latency, 50)), 1),
                               "p95": round(float(np.percentile(latency, 95)), 1)} if latency else None),
        "level_distribution": {lv: sum(r["report"]["level"] == lv for r in clean) for lv in LEVELS},
        "aggregate_score": ({"mean": round(float(np.mean(scores)), 3), "std": round(float(np.std(scores)), 3),
                             "min": round(min(scores), 3), "max": round(max(scores), 3)} if scores else None),
        "dimension_means": dimension_means(clean),
        "note": "spend excludes the embedding calls (text-embedding-3-small, negligible)",
        "errors": [r["error"] for r in recs if r["status"] == "error"][:5],
    }
    if labels is not None and clean:
        s["train_labels"] = agreement_with_labels(clean, labels)
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))
    return s


def _open_meta(out: Path, split: str, allow_unfrozen: bool, cache: bool) -> tuple[Path, dict]:
    """This run's teacher_meta, refusing an unfrozen teacher (unless allowed) and a label file
    another teacher started."""
    path = out.with_suffix(".teacher_meta.json")
    meta = teacher_meta.build(split, out.name)
    problems = teacher_meta.unfrozen(meta)
    if problems and not allow_unfrozen:
        sys.exit("Not a frozen teacher: " + "; ".join(problems) + ". Pass --allow-unfrozen for a trial run.")
    if out.exists():
        if not path.exists():
            sys.exit(f"{out.name} has no teacher_meta.json, so its teacher is unknown. Use a new --out.")
        old = json.loads(path.read_text(encoding="utf-8"))
        if diff := teacher_meta.differences(old, meta):
            sys.exit(f"{out.name} was labelled by a different teacher (differs in {', '.join(diff)}). Use a new --out.")
        old.setdefault("continued", []).append(meta["started"])
        meta = old
    else:
        meta.update(frozen=not problems, cache=cache)
    teacher_meta.write(path, meta)
    return path, meta


def label(args) -> None:
    ds = load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))[args.split]
    resumes, jobs = ds["resume_text"], ds["job_description_text"]
    out = TEACHER_LABELS / args.out
    meta_path, meta = _open_meta(out, args.split, args.allow_unfrozen, cache=not args.no_cache)

    # Extractions from another teacher version, model or host are never reused.
    checkout = meta["tag"] or (meta["commit"] or "no-git")[:12] + ("-dirty" if meta["dirty"] else "")
    version = "|".join([checkout, meta["models"]["extract"], ",".join(meta["providers"])])
    if args.no_cache:  # fresh extractions (each distinct text still extracted once per run)
        extractions, embeddings = ExtractionCache(Store(None, "extractions"), version), None
    else:
        extractions = ExtractionCache(Store(TEACHER_CACHE / "extractions.sqlite", "extractions"), version)
        embeddings = EmbeddingCache(Store(TEACHER_CACHE / "embeddings.sqlite", "embeddings"))
    teacher.install(embeddings)
    prices = meta["prices_usd_per_token"]

    done = {i for i, r in latest(read_jsonl(out)).items() if r["status"] in DONE}
    todo = [i for i in select_rows(resumes, jobs, args.limit) if i not in done]
    texts = sorted({("resume", resumes[i]) for i in todo} | {("job", jobs[i]) for i in todo},
                   key=lambda kt: (kt[0], sha256(kt[1])))
    missing = [(k, t) for k, t in texts if extractions.get(k, t) is None]
    print(f"{len(todo)} rows to label ({len(done)} done), {len(missing)} of {len(texts)} texts to extract -> {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    log, write_log = _appender(out.with_suffix(".extractions.jsonl"))
    labels, write_label = _appender(out)
    stopped = False
    try:
        run_parallel([partial(extract_text, k, t, extractions, prices.get(meta["models"]["extract"]))
                      for k, t in missing], args.workers, write_log,
                     lambda r: f"extract {r['kind']} {r['sha256'][:10]} {r['status']} {r['latency_s']}s")
        run_parallel([partial(score_row, i, resumes[i], jobs[i], extractions, prices.get(meta["models"]["judge"]))
                      for i in todo], args.workers, write_label,
                     lambda r: f"row {r['row_id']} {r['status']} {r['latency_s']}s")
        meta["finished"] = teacher_meta.now()
        teacher_meta.write(meta_path, meta)
    except (teacher.FatalAPIError, StopRun, KeyboardInterrupt) as e:
        print(f"\nStopped: {type(e).__name__}: {e}\nFinished rows are kept; rerun the same command to continue.")
        stopped = True
    finally:
        log.close()
        labels.close()
    summarize(out, args.split, ds["label"] if args.split == "train" else None)
    if stopped:
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--limit", type=int, default=200, help="rows of the seeded sample; 0 labels every row")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="label file name under data/ml/teacher_labels/")
    ap.add_argument("--no-cache", action="store_true",
                    help="fresh extractions and embeddings, for the test-retest runs")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="run without a frozen teacher (trial runs; recorded as frozen: false)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        labels = None
        if args.split == "train":
            labels = load_dataset(HF_DATASET, revision=HF_REVISION, cache_dir=str(HF_CACHE))["train"]["label"]
        summarize(TEACHER_LABELS / args.out, args.split, labels)
    else:
        label(args)


if __name__ == "__main__":
    main()
