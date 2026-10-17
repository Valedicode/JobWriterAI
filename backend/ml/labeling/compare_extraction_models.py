"""
Compare candidate teacher models on the teacher's resume extraction, before freezing the teacher.

Each model extracts the same resumes (HF train split, plus optional PDFs) twice, through the
teacher's own extraction (app prompt, structured output, the app's extraction checks). Per model:

- slips: what the extraction checks had to remove (bullets copied into a second entry, repeated
  grades, job titles repeated as the company), per resume
- not verbatim: share of extracted bullets/descriptions whose opening words aren't in the source
  text (extraction should copy text, not reword or invent it)
- coverage: share of the source's content lines that reach some extracted field
- stability: whether the two runs agree (identical result; same bullets under the same entries)
- failures, latency, and cost from the token counts and OpenRouter's catalogue prices

Run from backend/ (needs OPENROUTER_API_KEY; about runs x resumes x models calls):
  python -m ml.labeling.compare_extraction_models --models deepseek/deepseek-v4.1-flash,qwen/qwen3.8-flash
A model written as direct:openai/<model> is called on OpenAI's own API (needs OPENAI_API_KEY),
priced as its OpenRouter twin.
Results: data/ml/model_comparison/<timestamp>/ (results.jsonl, summary.json, summary.md)
"""
import argparse
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
import requests
from datasets import load_dataset

from ml.config import ARTIFACTS, HF_CACHE, HF_DATASET, SEED
from ml.labeling.resume_text_extractor import catalog_id, extract_resume

CATALOG_URL = "https://openrouter.ai/api/v1/models"
_KEY_CHARS = 40
_MIN_LINE_CHARS = 30
_SLICE = 12


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("-\n", "")
    return re.sub(r"[^0-9a-zäöüß]+", " ", text.casefold()).strip()


def _squash(text: str) -> str:
    """Letters and digits only. The dataset's resumes are scraped text with words glued
    together ("ProfileExpert"); without spaces a faithful copy still matches its source."""
    return _norm(text).replace(" ", "")


def _texts(cv: dict) -> list[str]:
    """The free-text lines an extraction produced: bullets, descriptions, the summary."""
    out = [cv.get("summary") or ""]
    for section, fields in (("experience", ("responsibilities",)), ("projects", ("description", "outcomes")),
                            ("leadership_activities", ("description", "highlights")), ("education", ("details",))):
        for entry in cv.get(section) or []:
            if not isinstance(entry, dict):
                continue
            for f in fields:
                value = entry.get(f)
                out.extend(value if isinstance(value, list) else [value])
    return [t for t in out if isinstance(t, str) and len(_norm(t)) >= _MIN_LINE_CHARS]


def _placements(cv: dict) -> set[tuple]:
    """(section, entry index, bullet) for every bullet: where each bullet was put."""
    found = set()
    for section, f in (("experience", "responsibilities"), ("leadership_activities", "highlights"),
                       ("projects", "outcomes")):
        for i, entry in enumerate(cv.get(section) or []):
            if isinstance(entry, dict):
                found.update((section, i, _norm(b)) for b in entry.get(f) or [] if isinstance(b, str) and b.strip())
    return found


def _values(value) -> list[str]:
    """Every string in an extraction, in order (field names left out)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _values(v)]
    if isinstance(value, list):
        return [s for v in value for s in _values(v)]
    return []


def quality(cv: dict, source: str) -> dict:
    """Verbatim share of the extracted lines, and coverage: the share of the source's
    12-character slices (spaces removed) found in the extracted values. Slices instead of
    lines or sentences, because the scraped texts have few line breaks and glue headings onto
    content ("Professional ProfileExpert in ...")."""
    src = _squash(source)
    lines = _texts(cv)
    verbatim = [_squash(t)[:_KEY_CHARS] in src for t in lines]
    extracted = "".join(_squash(v) for v in _values(cv))
    covered = [src[i:i + _SLICE] in extracted for i in range(0, len(src) - _SLICE + 1, _SLICE)]
    return {
        "lines": len(lines),
        "not_verbatim": round(1 - mean(verbatim), 3) if verbatim else 0.0,
        "coverage": round(mean(covered), 3) if covered else 1.0,
        "bullets": len(_placements(cv)),
    }


def load_prices(path: str | None) -> dict[str, tuple[float, float]]:
    """{model id: (USD per input token, USD per output token)} from OpenRouter's catalogue."""
    data = json.loads(Path(path).read_text(encoding="utf-8")) if path else requests.get(CATALOG_URL, timeout=180).json()
    prices = {}
    for m in data["data"]:
        p = m.get("pricing") or {}
        try:
            prices[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
        except (TypeError, ValueError):
            continue
    return prices


def pick_resumes(n: int, pdfs: list[str]) -> list[tuple[str, str]]:
    """(id, text): n distinct train resumes spread over the length range, plus the given PDFs."""
    ds = load_dataset(HF_DATASET, cache_dir=str(HF_CACHE))["train"]
    unique = sorted({t for t in ds["resume_text"] if t.strip()}, key=len)
    rng = np.random.default_rng(SEED)
    # One resume from each of n equal length bands, so short and long resumes are both tested.
    bands = np.array_split(np.arange(len(unique)), n)
    picks = [(f"hf-{int(i)}", unique[int(rng.choice(band))]) for i, band in enumerate(bands)]
    for path in pdfs:
        import pymupdf
        with pymupdf.open(path) as doc:
            picks.append((f"pdf-{Path(path).stem}", "\n".join(page.get_text("text") for page in doc)))
    return picks


def run_one(model: str, resume_id: str, text: str, run: int, prices: dict) -> dict:
    rec = {"model": model, "resume": resume_id, "run": run}
    t0 = time.perf_counter()
    try:
        out = extract_resume(text, model=model)
        price_in, price_out = prices.get(catalog_id(model), (0.0, 0.0))
        rec.update(status="ok", cv=out.cv, slips=len(out.notes), notes=out.notes,
                   input_tokens=out.input_tokens, output_tokens=out.output_tokens,
                   cost_usd=out.input_tokens * price_in + out.output_tokens * price_out, **quality(out.cv, text))
    except Exception as e:  # noqa: BLE001 - a failed call is a result for the comparison
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    rec["latency_s"] = round(time.perf_counter() - t0, 1)
    return rec


def summarize(records: list[dict], models: list[str], extractions_planned: int) -> dict:
    summary = {}
    for model in models:
        recs = [r for r in records if r["model"] == model]
        ok = [r for r in recs if r["status"] == "ok"]
        by_resume: dict[str, list[dict]] = {}
        for r in ok:
            by_resume.setdefault(r["resume"], []).append(r)
        pairs = [rs for rs in by_resume.values() if len(rs) >= 2]
        same = [json.dumps(a["cv"], sort_keys=True) == json.dumps(b["cv"], sort_keys=True) for a, b, *_ in pairs]
        jaccard = []
        for a, b, *_ in pairs:
            pa, pb = _placements(a["cv"]), _placements(b["cv"])
            jaccard.append(len(pa & pb) / len(pa | pb) if pa | pb else 1.0)
        summary[model] = {
            "extractions": len(recs),
            "failed": len(recs) - len(ok),
            "errors": sorted({r["error"][:120] for r in recs if r["status"] == "error"})[:3],
            "slips_per_resume": round(mean(r["slips"] for r in ok), 2) if ok else None,
            "resumes_with_slips": round(mean(r["slips"] > 0 for r in ok), 2) if ok else None,
            "not_verbatim": round(mean(r["not_verbatim"] for r in ok), 3) if ok else None,
            "coverage": round(mean(r["coverage"] for r in ok), 3) if ok else None,
            "identical_runs": round(mean(same), 2) if same else None,
            "same_bullet_placement": round(mean(jaccard), 3) if jaccard else None,
            "latency_s_median": round(float(np.median([r["latency_s"] for r in ok])), 1) if ok else None,
            "cost_per_extraction_usd": round(mean(r["cost_usd"] for r in ok), 5) if ok else None,
            "cost_full_extraction_usd": round(mean(r["cost_usd"] for r in ok) * extractions_planned, 2) if ok else None,
        }
    return summary


def to_markdown(summary: dict) -> str:
    cols = ["failed", "slips_per_resume", "resumes_with_slips", "not_verbatim", "coverage", "identical_runs",
            "same_bullet_placement", "latency_s_median", "cost_per_extraction_usd", "cost_full_extraction_usd"]
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for model, s in summary.items():
        lines.append(f"| {model} | " + " | ".join(str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    ap.add_argument("--resumes", type=int, default=12)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pdf", action="append", default=[], help="also extract this PDF CV (repeatable)")
    ap.add_argument("--catalog", help="a saved copy of the OpenRouter catalogue (else it is fetched)")
    # About 1,000 distinct texts in train + test (Section 5.3 of the plan): the full extraction.
    ap.add_argument("--full-extractions", type=int, default=1000)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prices = load_prices(args.catalog)
    unknown = [m for m in models if catalog_id(m) not in prices]
    if unknown:
        raise SystemExit(f"not in the OpenRouter catalogue: {unknown}")
    resumes = pick_resumes(args.resumes, args.pdf)
    out_dir = ARTIFACTS / "model_comparison" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(m, rid, text, run) for m in models for rid, text in resumes for run in range(1, args.runs + 1)]
    print(f"{len(tasks)} extractions ({len(models)} models x {len(resumes)} resumes x {args.runs} runs) -> {out_dir}")

    records = []
    with ThreadPoolExecutor(args.workers) as pool, (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        futures = [pool.submit(run_one, *task, prices) for task in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(tasks)}] {rec['model']} {rec['resume']} run {rec['run']}: {rec['status']} {rec['latency_s']}s")

    summary = summarize(records, models, args.full_extractions)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(to_markdown(summary) + "\n", encoding="utf-8")
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
