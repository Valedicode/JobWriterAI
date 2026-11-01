# Pre-registration

Distilling JobWriterAI's compatibility scorer into a small fine-tuned model (NLP course project).
Written 2026-10-24, before any model is trained and before the HF test labels are read. The git
commit that adds this file is the evidence of when it was fixed. Any analysis not listed here is
reported as exploratory, and any change after the commit is listed under Deviations with its
reason.

## What was seen before this was written

Stated so the reader can judge what could have shaped the choices below.

- **HF train labels** (all 6,234 deduplicated rows, including the rows now in `val`):
  - the teacher's agreement with them, ρ = 0.295 (NLP_PROJECT_PLAN.md, Appendix A.4)
  - the resume-prior diagnostic, ρ ≈ 0.28 (Appendix A.3)
  - the label counts of `train` and `val`
- **Teacher outputs** for every train and test row, summarised as score distributions only.
- **The teacher's test-retest agreement** on 100 train rows (Appendix A.4).
- **Not seen:** the HF test labels, and any student or baseline prediction.

## Frozen inputs

- **Teacher:** `teacher-v1` (`ad3a3e5`), labels in `data/ml/teacher_labels/{train,test}.jsonl`,
  each with its `teacher_meta.json`. Every row is clean.
- **Data:** HF `cnamuangtoun/resume-job-description-fit` at revision
  `08978e21714984bb417547d2c0f9b477f5298163`.
- **Split** (`ml/dataprep/splits.py`, seed 42): 34 of 280 train jobs form `val` (721 rows); the
  other 246 jobs form `train` (5,513 rows). `val` is used only for checkpoint selection, threshold
  fitting, prompt checks and the audit set. Nothing trains on it.
- **Test:** the full HF test split (1,759 rows, 71 jobs). 476 of its 477 resumes appear in the HF
  train split, so results generalise to unseen jobs only.

## Systems

| System | Trained on |
|---|---|
| Student-T 0.5B, 1.5B (Qwen2.5-Instruct, QLoRA) | teacher labels on `train` |
| Student-D 0.5B, 1.5B | dataset labels on the same `train` rows |
| Zero-shot Qwen2.5 0.5B, 1.5B (Student-T's prompt) | none |
| Zero-shot OpenAI model (direct API, dated id recorded) | none |
| Embedding cosine (`text-embedding-3-small`) | none |
| TF-IDF + ridge → teacher aggregate | teacher labels on `train` |
| TF-IDF + logistic regression → label | dataset labels on `train` |
| Resume prior: the resume's mean label over `train` | dataset labels on `train` |
| Majority class | dataset labels on `train` |

Training recipe as in NLP_PROJECT_PLAN.md §7.2: LoRA r 16, α 32, dropout 0.05 on all attention
and MLP projections; lr 2e-4, cosine, 3% warmup; up to 3 epochs; inputs truncated to 1,536 resume
and 768 job tokens; greedy decoding. Seeds: 42, 43, 44 for 0.5B; 42 (and 43 if compute allows) for
1.5B.

## Hypotheses

- **H1 (fidelity).** Student-T agrees with the teacher far better than any zero-shot model and
  better than TF-IDF + ridge. 1.5B ≥ 0.5B. It stays below the teacher's test-retest ceiling
  (ρ = 0.945).
- **H2 (validity).** Against the dataset labels, the teacher and Student-T are weak (ρ around 0.3).
  Student-D scores higher, and part of its advantage comes from memorising resumes, so it is close
  to the resume prior.
- **H3 (efficiency).** Per pair, the students are at least ten times faster and cheaper than the
  teacher pipeline, and the 0.5B student runs on CPU at a usable speed. Descriptive; not tested.

## Primary metrics

- **Fidelity:** Spearman ρ between a system's continuous score and the teacher's aggregate score.
- **Validity:** Spearman ρ between a system's continuous score and the ordinal dataset label
  (No Fit 0, Potential Fit 1, Good Fit 2).
- **Continuous scores:** Student-T and the zero-shot models give the generated `aggregate_score`;
  Student-D gives the expected ordinal value from the probabilities of the three labels' first
  tokens; the baselines give their prediction or similarity.

Both are computed on the HF test split, on every row. A row whose output can't be parsed gets the
system's mean score on `val`, so a system can't gain by failing. Coverage is reported beside it.

## Primary comparisons

Four, each a difference in ρ on test with a 95% CI from a paired, job-clustered bootstrap (resample
test jobs with replacement, 2,000 replicates, seed 42). p-values from the same bootstrap are
Holm-corrected across the four.

| # | Comparison | Metric | Predicted |
|---|---|---|---|
| C1 | Student-T 1.5B vs zero-shot Qwen2.5 1.5B | fidelity | Student-T higher |
| C2 | Student-T 1.5B vs TF-IDF + ridge | fidelity | Student-T higher |
| C3 | Student-D 1.5B vs Student-T 1.5B | validity | Student-D higher |
| C4 | Student-D 1.5B vs resume prior | validity | two-sided: does Student-D add anything beyond memorising resumes? |

Students in C1–C4 are the seed-42 runs; the other seeds give the seed sd reported beside them. If
the 1.5B runs can't be completed within the compute budget, the 0.5B students take their place in
C1–C4, and this is reported as a deviation.

## Secondary analyses (reported, not tested)

- MAE and Pearson against the teacher; per-dimension ρ and MAE; JSON validity rate; the
  consistency gap between the generated aggregate and the aggregate recomputed from the generated
  dimensions with the teacher's weights.
- Band agreement with the teacher's levels: accuracy and quadratic-weighted κ.
- Three-class validity: macro-F1 (main), QWK and accuracy after thresholds (below).
- Every system on `val` and on the author audit set; κ between the author and the dataset labels,
  and between the author and the teacher's band.
- 0.5B vs 1.5B, the seed sd, and the length strata (truncated vs untruncated inputs).
- Efficiency: latency median / p95, throughput, peak memory, $ per 1,000 pairs, the one-off
  labelling and training costs, and the break-even volume.

## Procedures fixed in advance

- **Checkpoint:** for each run, the epoch checkpoint with the lowest `val` loss.
- **Thresholds:** each system gets two thresholds on its continuous score, fitted on `val` to
  maximise macro-F1 over a grid of its `val` score quantiles (1st to 99th percentile), then applied
  unchanged to test. The teacher's four levels are not collapsed by hand.
- **Zero-shot prompts:** at most three prompt versions are tried on `val` (for parse rate only, not
  agreement); the last is frozen.
- **Exclusions:** rows the teacher couldn't label cleanly are excluded everywhere (there are none
  in `teacher-v1`'s labels). No other rows are excluded.
- **Opening the test labels:** once, after every checkpoint, prompt and threshold is frozen and
  every system's test predictions are written. The sha256 of each prediction file is committed
  before the labels are read.

## Deviations

None yet.
