"""How well the teacher agrees with itself: two label runs of the same rows (NLP_PROJECT_PLAN.md §5.1).

Both runs must use fresh extractions and embeddings (``--no-cache``); runs sharing a cache share most
of their work and would look more consistent than the teacher is. The agreement is the noise ceiling
for the students: a student cannot meaningfully agree with the teacher better than the teacher agrees
with itself.

Run from backend/, after the two runs in ml/README.md:
  python -m ml.labeling.teacher_retest                      # retest-a.jsonl vs retest-b.jsonl
  python -m ml.labeling.teacher_retest --a x.jsonl --b y.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ml.config import TEACHER_LABELS
from ml.labeling import teacher_meta
from ml.labeling.label_files import LEVELS, latest, read_jsonl, spearman


def clean_scores(path: Path) -> dict[int, dict[str, float | str]]:
    """Per clean row (ok, no flags): the aggregate score, each dimension's score and the level."""
    rows = {}
    for i, r in latest(read_jsonl(path)).items():
        if r["status"] == "ok" and not r["flags"]:
            report = r["report"]
            rows[i] = {"aggregate": report["aggregate_score"], "level": report["level"],
                       **{d["name"]: d["score"] for d in report["dimensions"]}}
    return rows


def icc_a1(x, y) -> float:
    """ICC(A,1), two-way random effects, absolute agreement of single measurements (McGraw & Wong
    1996). Unlike Spearman it also counts a shift between the runs as disagreement."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    grand = m.mean()
    ms_rows = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ms_err = ((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


def agreement(x, y) -> dict:
    diff = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return {"n": len(diff), "spearman": round(spearman(x, y), 3), "icc_a1": round(icc_a1(x, y), 3),
            "mean_abs_diff": round(float(diff.mean()), 4), "p95_abs_diff": round(float(np.percentile(diff, 95)), 4),
            "max_abs_diff": round(float(diff.max()), 4)}


def identical_extractions(a: Path, b: Path) -> dict:
    """Per kind, the share of texts whose extraction came out the same in both runs."""
    def values(path):
        return {(r["kind"], r["sha256"]): r["value"] for r in read_jsonl(path) if r["status"] == "ok"}
    va, vb = values(a.with_suffix(".extractions.jsonl")), values(b.with_suffix(".extractions.jsonl"))
    shared = va.keys() & vb.keys()
    out = {}
    for kind in ("resume", "job"):
        keys = [k for k in shared if k[0] == kind]
        if keys:
            out[kind] = {"texts": len(keys), "identical": round(sum(va[k] == vb[k] for k in keys) / len(keys), 3)}
    return out


def retest(a: Path, b: Path) -> dict:
    meta_a, meta_b = (json.loads(p.with_suffix(".teacher_meta.json").read_text(encoding="utf-8")) for p in (a, b))
    if diff := teacher_meta.differences(meta_a, meta_b):
        sys.exit(f"{a.name} and {b.name} come from different teachers (differ in {', '.join(diff)}).")
    cached = [p.name for p, m in ((a, meta_a), (b, meta_b)) if m.get("cache", True)]
    sa, sb = clean_scores(a), clean_scores(b)
    rows = sorted(sa.keys() & sb.keys())
    if len(rows) < 3:
        sys.exit(f"Only {len(rows)} rows are clean in both runs; nothing to compare.")
    quantities = ["aggregate"] + [q for q in sa[rows[0]] if q not in ("aggregate", "level")]
    levels_a, levels_b = [LEVELS.index(sa[i]["level"]) for i in rows], [LEVELS.index(sb[i]["level"]) for i in rows]
    s = {
        "runs": [a.name, b.name],
        "teacher": {"tag": meta_a["tag"], "providers": meta_a["providers"], "models": meta_a["models"]},
        "rows": {"a": len(sa), "b": len(sb), "compared": len(rows)},
        "warning": (f"{', '.join(cached)} used the teacher cache, so this overstates the agreement"
                    if cached else None),
        "agreement": {q: agreement([sa[i][q] for i in rows], [sb[i][q] for i in rows])
                      for q in quantities if all(q in sa[i] and q in sb[i] for i in rows)},
        "level": {"same": round(float(np.mean(np.equal(levels_a, levels_b))), 3),
                  "off_by_more_than_one": int(np.sum(np.abs(np.subtract(levels_a, levels_b)) > 1))},
        "identical_extractions": identical_extractions(a, b),
    }
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="retest-a.jsonl", help="first run, under teacher_labels/")
    ap.add_argument("--b", default="retest-b.jsonl", help="second run, under teacher_labels/")
    ap.add_argument("--out", default="retest.summary.json", help="written under teacher_labels/")
    args = ap.parse_args()
    s = retest(TEACHER_LABELS / args.a, TEACHER_LABELS / args.b)
    (TEACHER_LABELS / args.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
