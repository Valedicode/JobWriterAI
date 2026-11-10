"""The fidelity and validity tables of every system with a prediction file (ml/PREREGISTRATION.md).

- Fidelity: Spearman rho with the teacher's aggregate (plus Pearson, and MAE for scores on the
  teacher's scale).
- Validity: Spearman rho with the dataset label (0/1/2), and macro-F1 / QWK / accuracy after two
  thresholds fitted on `val`.
- 95% CIs from the job-clustered bootstrap; the primary comparisons C1-C4 as paired differences,
  Holm-corrected.
- A score that couldn't be parsed gets the system's `val` mean; coverage is reported beside it.

Run from backend/:
  python -m ml.evaluation.evaluate                     # the val table (any time)
  python -m ml.evaluation.evaluate --write-hashes      # Phase 6: record the prediction files, then commit
  python -m ml.evaluation.evaluate --split test --open-test-labels   # Phase 6, once
Output: data/ml/results/<split>.json and a table on stdout.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

from ml.config import ARTIFACTS, BACKEND_DIR
from ml.evaluation import predictions
from ml.evaluation.metrics import (LABELS, accuracy, bootstrap_ci, classify, fit_thresholds, holm, macro_f1, mae,
                                   paired_difference, pearson, qwk, safe_spearman)

RESULTS = ARTIFACTS / "results"
FROZEN_HASHES = BACKEND_DIR / "ml" / "TEST_PREDICTIONS.sha256"
# (id, system a, system b, metric). {model} is the 1.5B student's model, or 0.5B as the
# pre-registered fallback.
COMPARISONS = [
    ("C1", "student-t-{model}-seed42", "zeroshot-{model}", "fidelity"),
    ("C2", "student-t-{model}-seed42", "tfidf-ridge", "fidelity"),
    ("C3", "student-d-{model}-seed42", "student-t-{model}-seed42", "validity"),
    ("C4", "student-d-{model}-seed42", "resume-prior", "validity"),
]


def aligned(prediction: dict, split: str, ids: list[int]) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Scores in reference order, unparsed ones replaced by the system's `val` mean; the predicted
    classes when the system gives them; and the share of rows parsed."""
    by_split = {s: {r["row_id"]: r for r in prediction["rows"] if r["split"] == s} for s in predictions.SPLITS}
    val_scores = [r["score"] for r in by_split["val"].values() if r["score"] is not None]
    fill = float(np.mean(val_scores)) if val_scores else 0.0
    rows = [by_split[split].get(i) for i in ids]
    parsed = [r is not None and r["score"] is not None for r in rows]
    scores = np.array([r["score"] if ok else fill for r, ok in zip(rows, parsed)], dtype=float)
    classes = np.array([r["label"] for r in rows]) if all(r is not None and "label" in r for r in rows) else None
    return scores, classes, float(np.mean(parsed))


def val_thresholds(prediction: dict, val_ref: dict) -> tuple[float, float]:
    scores, _, _ = aligned(prediction, "val", val_ref["row_id"])
    return fit_thresholds(scores, val_ref["label"])


def evaluate_system(name: str, prediction: dict, ref: dict, val_ref: dict, split: str) -> dict:
    scores, classes, coverage = aligned(prediction, split, ref["row_id"])
    teacher, label, jobs = np.array(ref["teacher"]), np.array(ref["label"]), ref["job"]
    out = {"coverage": coverage, "scale": prediction["scale"]}
    if name != "teacher":
        out["fidelity"] = {"rho": safe_spearman(scores, teacher),
                           "ci": bootstrap_ci(lambda r: safe_spearman(scores[r], teacher[r]), jobs),
                           "pearson": pearson(scores, teacher)}
        if prediction["scale"] == "teacher":
            out["fidelity"]["mae"] = mae(scores, teacher)
    if classes is None:
        thresholds = val_thresholds(prediction, val_ref)
        classes = classify(scores, thresholds)
        out["thresholds"] = thresholds
    out["validity"] = {"rho": safe_spearman(scores, label),
                       "ci": bootstrap_ci(lambda r: safe_spearman(scores[r], label[r]), jobs),
                       "macro_f1": macro_f1(label, classes), "qwk": qwk(label, classes),
                       "accuracy": accuracy(label, classes)}
    return out


def comparisons(systems: dict, ref: dict, split: str) -> list[dict]:
    model = next((m for m in ("Qwen2.5-1.5B-Instruct", "Qwen2.5-0.5B-Instruct")
                  if f"student-t-{m}-seed42" in systems), None)
    if model is None:
        return []
    target = {"fidelity": np.array(ref["teacher"]), "validity": np.array(ref["label"])}
    out = []
    for cid, a, b, metric in COMPARISONS:
        a, b = a.format(model=model), b.format(model=model)
        if a not in systems or b not in systems:
            out.append({"id": cid, "a": a, "b": b, "metric": metric, "missing": True})
            continue
        sa, _, _ = aligned(systems[a], split, ref["row_id"])
        sb, _, _ = aligned(systems[b], split, ref["row_id"])
        y = target[metric]
        result = paired_difference(lambda r: safe_spearman(sa[r], y[r]), lambda r: safe_spearman(sb[r], y[r]),
                                   ref["job"])
        out.append({"id": cid, "a": a, "b": b, "metric": metric, **result})
    tested = [c for c in out if "p" in c]
    for c, p in zip(tested, holm([c["p"] for c in tested])):
        c["p_holm"] = p
    return out


def frozen_hashes() -> dict[str, str]:
    if not FROZEN_HASHES.exists():
        return {}
    pairs = (line.split() for line in FROZEN_HASHES.read_text(encoding="utf-8").splitlines() if line.strip())
    return {system: digest for digest, system in pairs}


def committed(path) -> bool:
    run = lambda *a: subprocess.run(["git", *a], cwd=BACKEND_DIR, capture_output=True).returncode == 0
    return run("ls-files", "--error-unmatch", str(path)) and run("diff", "--quiet", "HEAD", "--", str(path))


def check_frozen(systems: dict) -> None:
    """The test labels are opened only once every prediction file is recorded in a commit."""
    recorded = frozen_hashes()
    problems = [f"{s}: {'not recorded' if s not in recorded else 'changed since recorded'}"
                for s in systems if recorded.get(s) != predictions.sha256_of(s)]
    if not committed(FROZEN_HASHES):
        problems.append(f"{FROZEN_HASHES.name} is missing or not committed")
    if problems:
        raise SystemExit("Not opening the test labels:\n  " + "\n  ".join(problems)
                         + "\nRun --write-hashes, check the list, commit it, then run again.")


def fmt(x, digits=3) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def table(results: dict, split: str) -> str:
    lines = [f"## {split}", "",
             "| System | Coverage | Fidelity ρ [95% CI] | MAE | Validity ρ [95% CI] | Macro-F1 | QWK | Acc |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in results["systems"].items():
        f, v = r.get("fidelity", {}), r["validity"]
        fid = f"{fmt(f.get('rho'))} [{fmt(f['ci'][0])}, {fmt(f['ci'][1])}]" if f else "—"
        val = f"{fmt(v['rho'])} [{fmt(v['ci'][0])}, {fmt(v['ci'][1])}]"
        lines.append(f"| {name} | {r['coverage']:.0%} | {fid} | {fmt(f.get('mae'))} | {val} | "
                     f"{fmt(v['macro_f1'])} | {fmt(v['qwk'])} | {fmt(v['accuracy'])} |")
    for c in results["comparisons"]:
        if c.get("missing"):
            lines.append(f"\n{c['id']}: {c['a']} vs {c['b']} ({c['metric']}) not run, a system is missing")
        else:
            lines.append(f"\n{c['id']}: {c['a']} − {c['b']} ({c['metric']} ρ) = {c['difference']:+.3f} "
                         f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], p = {c['p']:.4f}, Holm p = {c['p_holm']:.4f}")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console's default codec has no ρ
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=predictions.SPLITS, default="val")
    ap.add_argument("--open-test-labels", action="store_true", help="Phase 6 only: read the HF test labels")
    ap.add_argument("--write-hashes", action="store_true", help="record the prediction files' sha256")
    args = ap.parse_args()

    systems = predictions.read_all()
    if not systems:
        raise SystemExit(f"No prediction files in {predictions.PREDICTIONS}; run ml.evaluation.baselines first.")
    if args.write_hashes:
        FROZEN_HASHES.write_text("".join(f"{predictions.sha256_of(s)}  {s}\n" for s in systems), encoding="utf-8")
        print(f"Recorded {len(systems)} prediction files in {FROZEN_HASHES}; commit it before --split test.")
        return
    if args.split == "test":
        if not args.open_test_labels:
            raise SystemExit("The test labels are opened once, in Phase 6: pass --open-test-labels.")
        check_frozen(systems)

    val_ref = predictions.reference("val", with_labels=True)
    ref = val_ref if args.split == "val" else predictions.reference("test", with_labels=True)
    results = {"split": args.split, "rows": len(ref["row_id"]), "labels": list(LABELS),
               "systems": {name: evaluate_system(name, p, ref, val_ref, args.split) for name, p in systems.items()}}
    results["comparisons"] = comparisons(systems, ref, args.split)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.split}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(table(results, args.split))
    print(f"\n-> {RESULTS / (args.split + '.json')}")


if __name__ == "__main__":
    main()
