# NLP Course Project: Distilling JobWriterAI's Compatibility Scorer into a Small Fine-Tuned Model

Plan v2.1, 2026-10-17. Supersedes the 2026-09-17 draft (v1). The proposal was approved on 2026-10-17.

v2 keeps v1's scope and adds three things: what the data already in hand says (Section 2), a sturdier
evaluation protocol (Sections 3, 8-11) and a phased plan with the current status (Section 13).

v2.1 updates the status after the 2026-10-17 app fixes were merged, records what they mean for the
teacher, and sets the order of distillation work versus app features (Section 13.0).

---

## 0. Status (2026-10-17)

| Area | State |
|---|---|
| Proposal | Approved 2026-10-17 |
| App fixes | Merged into `develop` on 2026-10-17: the code-review fixes (#62–#68) and the base-app fixes (#69–#74). See "What the merged app fixes mean for the teacher" below |
| Scaffolding: `backend/ml/` config, requirements, labelling, training script with a CPU smoke test | Committed on `feature/distillation` (`8f91d91`, 2026-09-18). `develop` was merged in locally (`05900e3`, up to #68, not pushed); #69–#74 still need merging in |
| Phase 0 work in progress | Parked in `git stash` on `feature/distillation` ("phase0-wip"): dataset script moved to `backend/ml/dataprep/`, scheduler fix, extractor reads the app's prompt and uses the routed model, pinned requirements. The CPU smoke test hasn't run on it yet |
| HF dataset prep and due-diligence report | Done (Appendix A). The script is committed only once the parked Phase 0 work lands (it lived in the ignored `backend/ml/data/`) |
| Teacher pilot | 200 train rows, run on the 2026-09-17 scorer. Outdated: the scorer has changed 13 times since (rubric, knockouts, eligibility, experience date ranges, model routing), and the CV extraction changed with #71 |
| Teacher frozen (tagged) | No. Nothing in `develop` blocks the tag now; it waits only for the teacher-model decision (Decisions needed, 4) |
| Evaluation harness, baselines, inference wrapper, latency/cost harness, failure analysis | Not started |
| Report, presentation | Not started |

Problems found in the 2026-10-17 review, all scheduled in Phase 0/1 (the first three are fixed in
the parked Phase 0 work):

- `training/train_lora.py`: `lr_scheduler_type="cosine", warmup_steps=0.03` are inside a comment
  on the `learning_rate` line, so training runs at a constant learning rate with no warmup.
- `labeling/resume_text_extractor.py`:
  - Its copy of the CV-extraction prompt is stale. The app's prompt has since added rules for
    spoken languages, skill names without category labels, and contact fields.
  - It calls `ChatOpenAI("gpt-4o-mini")` directly instead of the app's routed
    `chat_model("extract")`, so resume and job extraction can run on different providers.
- `labeling/generate_teacher_labels.py` has none of the following:
  - an extraction cache (it makes 2 LLM extractions per row although the train split has only
    642 distinct resumes and 280 distinct jobs)
  - an embedding cache
  - retry/backoff
- Labelling cost is read from `get_openai_callback`, which may not price OpenRouter model ids.
- `requirements-ml.txt` uses `>=` ranges instead of pinned versions.

**What the merged app fixes mean for the teacher.** All of #62–#74 are pre-existing-app
maintenance and go in the report's disclosure table as such, never as course work.

- **Only #71 changes the teacher's input.** The app's CV extraction got new prompt rules and
  deterministic checks against the PDF text (`app/services/cv_extraction_checks.py`):
  - bullets copied into a second entry are removed
  - grade repeats are dropped
  - a company that only repeats the job title is cleared

  The teacher's text extractor must do the same: read the current prompt (the parked work already
  does) and run `clean_extracted_cv(result, text)`. Otherwise the labels come from a different
  extraction than the app's.
- **None of #62–#74 changes `scoring_agent.py`.** The rest touch cover letters, page count, the
  CV review, the editor and chat; they don't affect the teacher.
- **The editor refactor (`refactor/cv-view-segments`) keeps the printed CV byte-identical,**
  pinned by `tests/golden/cv_tex.json`, so it doesn't affect the teacher either.

---

## 1. Scope and attribution (read this before writing anything else)

This document plans **new work only**. It does not claim any part of the existing JobWriterAI
application as a contribution of this course project. That includes
`backend/app/agents/scoring_agent.py`, its skill-matching cascade,
`backend/app/data/skill_graph.json`, the extraction agents and the LangGraph orchestrator. That
code predates the course and is treated strictly as:

1. **The teacher** whose behaviour we distil: a black box we call, not code we wrote for this
   project.
2. **Reused infrastructure**: its Pydantic schemas and extraction tools, which the data-preparation
   scripts call like any third-party library.

Everything under `backend/ml/` is the course contribution:
- the labelling harness
- dataset preparation
- LoRA/QLoRA training
- the evaluation protocol and harness
- the analysis

In the report, describe `scoring_agent.py` in a "Background / Existing System" subsection, never
under "Method" or "Contributions".

Two further rules:

- **App maintenance is not course work.** Changes to the app made during the course are listed as
  pre-existing-app maintenance with their commit ranges. This covers the 2026-10-17 bug fixes and
  any teacher-affecting fix.
- **The teacher is one tagged version** (`teacher-v1`, Section 5.1). Labels from any other version
  are not mixed in.

The report's first page carries a prior-work vs. new-work table:

| | Prior work (JobWriterAI, pre-course) | New work (this project) |
|---|---|---|
| Code | `backend/app/**`, `frontend/**` | `backend/ml/**` |
| Commits | before 2026-09-17, plus listed app-maintenance commits | `feature/distillation` from `8f91d91` (2026-09-18) |
| Role | teacher, schemas, extraction | labels, students, baselines, evaluation, analysis |

---

## 2. What the data in hand already says (why v2 changes the protocol)

These checks use only the HF **train** split and the v0 pilot. They made no API calls, and the
test labels were not read.

1. **The (old) teacher barely tracks the dataset labels.** On the 200 pilot rows:
   - Spearman ρ between the teacher's aggregate score and the ordinal label
     (No Fit = 0, Potential Fit = 1, Good Fit = 2) is **0.16** (p = 0.02).
   - Mean aggregate by label: No Fit 0.36, Potential Fit 0.42, Good Fit 0.39. Good Fit scores
     *lower* than Potential Fit.
   - Per dimension: domain 0.26, hard_skills 0.20, experience 0.05, seniority −0.06,
     ats_keywords −0.04.
2. **The HF test split rewards memorising resumes.** Within train, hold out 20% of jobs (1,282
   rows; 99% of them have their resume elsewhere in train). Predicting each held-out row only from
   its resume's mean training label, without reading the job, gives **ρ = 0.28**, macro-F1 0.29.
   The test split has the same structure: 476 of its 477 resumes appear in train. Any model
   trained on dataset labels can exploit this; the teacher cannot. Labels vary within a resume
   (only 19% of resumes have one label across all their rows), so the prior is informative but far
   from deterministic.
3. **The dataset has no card, no licence and no description.** Checked on the Hugging Face API,
   2026-10-17: only `train.csv` and `test.csv`, created 2024-07-25. How the labels were made
   (human, LLM or heuristic) is unknown, and redistribution rights are unclear.
4. **The teacher's levels are compressed.** The pilot had 146 low / 53 medium / 1 high /
   0 excellent (bands: low < 0.45 ≤ medium < 0.65 ≤ high < 0.80 ≤ excellent). A 4-class level
   metric is effectively binary.

What v2 does about it:

- **The dataset labels are "reference labels of unknown provenance"**, never "human ground
  truth". A small author-labelled audit set (Section 6) estimates how noisy they are.
- **Fidelity and validity are measured separately**, each with its own primary metric.
  Fidelity is the student against the teacher; validity is any system against the reference
  labels. Continuous scores are primary and levels secondary.
- **A resume-prior baseline sits next to every supervised result**, and the generalisation claim
  is limited to unseen jobs.
- **The teacher's own test-retest agreement is measured** and used as the ceiling for fidelity.
- **The teacher is frozen and re-piloted before full labelling.** The numbers above come from an
  older scorer and may change.

The project stays meaningful whatever the re-pilot shows. Fidelity and cost are distillation
questions that don't depend on the teacher's validity. The shared-subset ablation (RQ3) measures
directly what the teacher is worth as a supervisor compared with the dataset's own labels.

Suggested framing for the report: *distilling a hybrid symbolic + LLM pipeline. What transfers to
a small model, at what cost, and whether validity is inherited.*

---

## 3. Research questions and hypotheses

If the approved proposal words the RQs differently, keep its wording and map the analyses below
onto it.

- **RQ1, fidelity.** How closely does a LoRA-tuned 0.5B / 1.5B model reproduce the teacher's
  scores (aggregate, the five dimensions, band) from raw text on unseen jobs? The comparison point
  is the teacher's own test-retest agreement.
- **RQ2, do we need to tune?** Does the tuned student beat cheaper alternatives, on fidelity and
  on agreement with the reference labels, and at what cost per pair? The alternatives are:
  - zero-shot prompting of the same base model
  - zero-shot prompting of an API model
  - embedding cosine
  - TF-IDF models
- **RQ3, label source (shared-subset ablation).** Train one student on teacher labels and one on
  the dataset's labels, using identical training pairs, recipe and budget. How do they differ in
  agreement with the reference labels and with the teacher?

Directional hypotheses. They are pre-registered in Phase 2, after the `teacher-v1` re-pilot.

- **H1:**
  - Tuned students reach much higher fidelity than zero-shot models.
  - 1.5B ≥ 0.5B.
  - The gap to the teacher's test-retest ceiling is reported, not assumed.
- **H2:**
  - Against the reference labels, the teacher and the teacher-trained students are weak
    (v0 pilot ρ ≈ 0.16).
  - Students trained on dataset labels score higher, partly through resume memorisation
    (resume prior ≈ 0.28).
- **H3:**
  - Per pair, students are at least an order of magnitude faster and cheaper than the teacher
    pipeline.
  - A 0.5B student is usable on CPU, which answers the GPU-deployment question.

---

## 4. Repo layout

New, isolated module; nothing is added to `backend/app/agents/`. Generated artefacts go to
`backend/data/ml/`, which is git-ignored.

```
backend/ml/
  README.md                        # exact reproduction steps (reproducibility bonus)
  PREREGISTRATION.md               # committed before the test labels are opened (Phase 2)
  config.py                        # paths, seed, model ids, price table, label maps
  requirements-ml.txt              # pinned versions
  datasets/                        # renamed from data/: backend/.gitignore ignores every data/ folder
    prepare_hf_eval_dataset.py     # download, dedupe, sha256 of the CSVs, length report
    splits.py                      # job-grouped train/val split, stable row ids
    audit_sample.py                # draws the manual audit sample from val
  labeling/
    resume_text_extractor.py       # imports the app's CV-extraction prompt; uses chat_model("extract")
    generate_teacher_labels.py     # resumable, cached, retried; flags; cost from tokens
    caches.py                      # sqlite: extraction by text sha256, embeddings by (text, model)
    teacher_meta.py                # writes teacher_meta.json (tag, commit, models, versions, dates)
  training/
    train_lora.py                  # Student-T and Student-D (--target teacher|dataset)
  evaluation/
    run_inference.py               # students and zero-shot models; JSON parsing with failure counts
    baselines.py                   # embedding cosine, TF-IDF + ridge/LR, resume prior, majority
    metrics.py                     # Spearman, MAE, QWK, macro-F1, fitted thresholds, clustered bootstrap
    eval_fidelity.py
    eval_external.py
    eval_latency_cost.py
    failure_analysis.py
  notebooks/
    train_kaggle.ipynb             # thin wrapper around training/ and evaluation/
  tests/                           # unit tests for metrics, splits, parsing, caches

backend/data/ml/                   # git-ignored: HF cache, labels, caches, checkpoints, predictions, results
```

---

## 5. Teacher

### 5.1 Definition and freeze

The teacher has two stages:
- **Extraction:** jobs go through the app's `retrieve_text`. Resumes go through a text variant of
  the `cv_agent` extraction, which **imports** the app's prompt instead of copying it.
- **Scoring:** `_calculate_compatibility_score_v2_internal`.

It is pinned to the git tag `teacher-v1` on `develop`, set after the 2026-10-17 fixes are merged.

`teacher_meta.json` sits next to the labels and records:
- tag and commit
- provider (OpenRouter or OpenAI)
- model per role (`LLM_MODEL_EXTRACT`, `LLM_MODEL_JUDGE`, or the shared `LLM_MODEL`)
- embedding model and temperature
- package versions
- labelling start and end dates

The report copies these into the Data section.

No scorer changes after the tag. A teacher bug found later is documented as a limitation. If it
must be fixed, everything is relabelled under `teacher-v2`.

### 5.2 Labelling pipeline requirements

- **Caching.** Extraction is cached by the sha256 of the text, so the train split needs 642 resume
  and 280 job extractions instead of 2 × 6,241. Embeddings use a persistent on-disk cache.
- **Robustness.** Exponential backoff on rate limits and timeouts. Resumable, keyed by row id,
  flushing per row.
- **Flags.** Flagged rows are excluded from training and counted in the report:
  - LLM transferability unjudged
  - domain embedding failed
  - degenerate CV
- **Recorded but kept:**
  - experience fallback ("Could not estimate candidate years")
  - level capped by an unmet knockout
- **Cost** is computed from token counts × a price table in `config.py` and cross-checked against
  the provider dashboard, not read from `get_openai_callback`.

### 5.3 Runs

1. **Re-pilot.** Run the same 200 seeded rows on `teacher-v1`. Compare with v0 (Appendix A):
   - level distribution
   - flag rates
   - cost and latency
   - ρ against the train labels
2. **Test-retest.** Label 100 random train rows twice with all caches off. Report
   Spearman / ICC(2,1) on the aggregate and each dimension, plus band agreement. This is the
   fidelity ceiling for RQ1.
3. **Full run.** All deduplicated train pairs (~6,234) and all 1,759 test pairs. For test, the
   teacher sees only the texts; the labels stay sealed. Running the full test split rather than a
   sample costs little because its 1,759 pairs have only ~548 distinct texts, and it means every
   system is scored on the same rows.
   - **Cost:** ≈ $14 for ~8k pairs at v0's per-row cost, less with extraction caching. The
     re-pilot gives the real figure under `teacher-v1`'s models.
   - **Wall-clock:** ~20 h at v0's rate (200 rows in ~30 min with 8 workers). Caching and 16
     workers should bring it under a day; run it overnight.

---

## 6. Data and splits

- **HF train** (6,241 rows). Drop the 7 duplicate pairs, then split **by job**: ~12% of jobs
  (~34 jobs, ~750 rows) form `val` and the rest is `train`. `val` is used only for:
  - checkpoint selection
  - fitting the 3-class thresholds
  - prompt checks
  - the audit set

  Nothing trains on `val`.
- **HF test** (1,759 rows) is for the final evaluation only, opened once (Section 9). It measures
  generalisation to unseen jobs, not unseen candidates.
- **Optional unseen-resume diagnostic.** Hold out ~15% of resumes and ~15% of jobs from train and
  evaluate on the pairs where both are held out. This says something about new candidates, which
  the HF test split cannot.
- **Manual audit set.**
  - 80 pairs from `val`, stratified across its jobs.
  - The author labels them blind on the same three classes, following a written rubric, before
    seeing the dataset labels or teacher scores.
  - If a second annotator is available, label 30 overlapping pairs to get inter-annotator κ.
  - Report κ(author, dataset labels), κ(author, teacher band at fitted thresholds) and ρ of every
    system against the author labels.
  - This puts a noise estimate on the reference labels and gives a second, small reference set.
- **Licensing.** The dataset has no licence, so use it for research only:
  - Publish code, row ids and numeric outputs, never resume or job text.
  - Record the sha256 of `train.csv` / `test.csv` and the download date, in case the dataset
    changes or disappears.

---

## 7. Students

### 7.1 Input and targets

- **Input:** raw resume and job text, truncated to 1,536 / 768 Qwen tokens (about p90). Report the
  share of truncated rows, and stratify errors by length.
- **Student-T target.** Dimensions come first, so the aggregate is generated after them, mirroring
  how the teacher computes it:
  ```json
  {"dimensions": {"hard_skills": 0.32, "experience": 0.50, "seniority": 0.60, "domain": 0.41, "ats_keywords": 0.18},
   "aggregate_score": 0.38, "level": "low"}
  ```
  Values are rounded to 2 decimals. The gap-analysis skill lists are not a target: generating open
  vocabulary has no clean evaluation story at this model size. That stays a discussion item.
- **Student-D target:** `{"label": "No Fit" | "Potential Fit" | "Good Fit"}`. Its continuous score
  is the expected ordinal value from the probabilities of the three labels' first tokens.
- **Shared subset.** Student-T and Student-D train on the **same pairs**: the clean
  teacher-labelled pairs in `train`. They share the recipe, steps and seeds. Only the label source
  differs.
- **Derived checks for Student-T:**
  - Compare the generated `level` token with the band of the generated aggregate.
  - Recompute the aggregate from the predicted dimensions with the teacher's weights
    (0.40 / 0.15 / 0.10 / 0.10 / 0.10, normalised), and report the gap to the generated aggregate
    as a consistency check.

  The teacher's level is the band of its aggregate, except when an unmet knockout caps it at
  medium. That case is recorded in the labels.

### 7.2 Recipe

- **Models:** `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`.
- **Adapters:** QLoRA (nf4) on GPU. LoRA r = 16, α = 32, dropout 0.05, on all attention and MLP
  projections.
- **Loss:** completion-only, so prompt tokens are masked (already checked by the smoke test).
- **Optimisation:** lr 2e-4 with a cosine schedule and 3% warmup (fix the commented-out
  arguments). 2-3 epochs; the checkpoint is chosen on `val` loss.
- **Decoding:** greedy at evaluation, with a fixed `max_new_tokens` (96 for Student-T, 16 for
  Student-D).
- **Seeds:** 3 for every 0.5B configuration, and 1 (2 if quota allows) for 1.5B. Report mean ± sd.
- **Stretch, in order:**
  1. aggregate-first vs dimensions-first target
  2. learning curve at 25 / 50 / 100% of the labels
  3. Qwen2.5-3B
  4. a student trained on both label sources

### 7.3 Compute (Kaggle, ~30 GPU-h/week, T4 16 GB)

These are estimates, to be replaced by the smoke-test timings.

- **Tokens:** ~5.5k examples × ~1.6k tokens ≈ 9M tokens per epoch.
- **Training:** 1.5B ≈ 1-1.7 h per epoch; 0.5B ≈ 0.3-0.6 h per epoch.
- **Inference** on val + test (~2.5k pairs): ≈ 20-40 min per model with batching.
- **Total:** all core runs (2 × 1.5B, 6 × 0.5B, zero-shot inference) ≈ 15-20 GPU-h, about one
  Kaggle week.

Run a 200-example, 1-epoch smoke test before every new configuration.

---

## 8. Systems compared (one harness, identical rows)

| System | Trained on | Fidelity (vs teacher) | Validity (vs reference labels) |
|---|---|---|---|
| Teacher (cascade) | n/a | test-retest ceiling | yes |
| Student-T 0.5B / 1.5B | teacher labels | yes | yes |
| Student-D 0.5B / 1.5B | dataset labels | yes (expected poor) | yes |
| Zero-shot Qwen 0.5B / 1.5B (same prompt as Student-T) | n/a | yes | yes |
| Zero-shot API model (`gpt-4o-mini`) | n/a | yes | yes |
| Embedding cosine (`text-embedding-3-small`, the teacher's model) | n/a | yes | yes |
| TF-IDF + ridge → teacher aggregate | teacher labels | yes | yes |
| TF-IDF + logistic regression → label | dataset labels | n/a | yes |
| Resume prior (resume's mean train label) | dataset labels | n/a | yes (leakage reference) |
| Majority class | dataset labels | n/a | yes |

- **TF-IDF features:** TF-IDF vectors of the resume and the job (fitted on `train`), their cosine,
  and their element-wise product, so the model sees the overlap of the pair rather than each text
  alone.
- **Zero-shot prompt:** the same system prompt and JSON schema as Student-T, plus a short
  description of the five dimensions and the 0-1 scale.
  - Parse failures are retried once, then counted.
  - Metrics are reported on parsed rows with coverage, and with failures imputed by the `val`
    mean.
  - Optional: a direct 3-class prompt for the API model, scored on validity only.

---

## 9. Metrics, statistics and pre-registration

**Fidelity (RQ1, RQ2):**
- **Primary:** Spearman ρ between predicted and teacher aggregate.
- **Secondary:**
  - MAE and Pearson
  - per-dimension MAE / ρ
  - band agreement (accuracy, quadratic-weighted κ)
  - JSON validity rate
  - the consistency gap (Section 7.1)
- **Sets:** `val` (unseen jobs), and the HF test texts scored against the teacher's outputs.

**Validity (RQ2, RQ3):**
- **Primary:** Spearman ρ between the continuous score and the ordinal reference label (0/1/2).
  This needs no label mapping.
- **Secondary:** macro-F1, QWK and accuracy after mapping scores to three classes.
  - Each system gets two thresholds, fitted on `val` to maximise macro-F1, then applied unchanged
    to test.
  - The teacher's 4-band levels are not collapsed by hand.
- **Also reported:** the same metrics on the author audit set.

**Efficiency (RQ2, H3):**
- latency median / p95 per pair (warm, same machine)
- throughput with batching
- peak memory
- $ per 1k pairs:
  - API token prices for the teacher and zero-shot
  - an equivalent on-demand T4 price for students on GPU
  - CPU for the 0.5B student
- **One-off costs:** labelling $ and training GPU-h.
- **Break-even volume:** the number of pairs after which the student is cheaper than calling the
  teacher. This answers the supervisor's training-vs-API question.

**Uncertainty:**
- 95% CIs from a **job-clustered bootstrap**: resample jobs, 2,000 replicates. Rows share jobs
  (~22 per job in train, ~25 in test), so resampling rows would understate the variance.
- Paired clustered bootstrap for the pre-registered comparisons.
- Seed sd for students.

**Pre-registration.** Phase 2 commits `backend/ml/PREREGISTRATION.md` (the git timestamp is the
evidence). It contains:
- the hypotheses
- the primary metric per RQ
- the short list of primary comparisons
- the exclusion rules
- the threshold procedure
- the checkpoint-selection rule

The HF test labels are opened **once**, after every checkpoint, prompt and threshold is frozen.
Any analysis not in the file is reported as exploratory.

---

## 10. Failure analysis (graded; schedule it, don't leave it for the end)

- **Largest disagreements:** the biggest |student − teacher| cases on `val` and test, put into
  categories such as:
  - truncated evidence
  - seniority mismatch
  - keyword overlap without the skill
  - knockout cap
  - experience fallback
- **Per-dimension fidelity:** which teacher components transfer. Compare the embedding-like
  `domain` with rule-based `seniority` / `experience` and with skill matching (`hard_skills`).
- **Length strata:** errors on truncated vs. untruncated inputs.
- **Reference-label disagreements:** cases where teacher and student agree with each other but
  disagree with the reference label. Cross-check them against the audit set: label noise or a
  teacher blind spot?
- **Privacy:** examples in the report are paraphrased. No verbatim resume text.

---

## 11. Threats to validity

| Threat | Handling |
|---|---|
| Reference labels of unknown provenance | Author audit set with κ; never called ground truth |
| Resume overlap between train and test | Claim unseen jobs only; resume-prior baseline beside every supervised result |
| Teacher non-determinism (LLM extraction and judge) | Test-retest ceiling; tagged teacher; recorded models |
| Teacher drift | `teacher-v1` tag; no scorer edits during labelling |
| Truncated inputs (student) vs. full text (teacher) | Share truncated; length-stratified errors |
| Domain shift: the app targets German/EU student CVs, the dataset is general-market English resumes | Fidelity results transfer; validity results are about this dataset only |
| Compressed teacher levels | Continuous metrics primary; levels secondary |
| Many comparisons | Few pre-registered primary comparisons; the rest exploratory |
| Small `val` (~34 jobs) | Clustered CIs; thresholds are its only fitted parameters |

---

## 12. Ethics, disclosure, reproducibility

- **Prior vs. new work:** the table in Section 1 goes on the report's first page. The
  pre-existing app is described as background.
- **AI assistance:** disclose it according to the course policy. This is separate from the
  JobWriterAI disclosure.
- **Dataset:** no licence, so research use only and no redistribution of text. Resumes may contain
  personal data, so no verbatim examples.
- **Reproducibility:**
  - `backend/ml/README.md` with exact commands
  - pinned `requirements-ml.txt`
  - fixed seeds
  - `teacher_meta.json`
  - dataset hashes
  - a link to the Kaggle notebook
  - released numeric predictions and teacher outputs keyed by row id, so the tables can be rebuilt
    without paying for relabelling
  - one script that rebuilds every results table from the saved predictions

---

## 13. Plan and status

Estimates are working days for one person, part-time. **Deadline: 2026-12-10.**

### Schedule (dated 2026-10-17)

| Week | Distillation | Milestone | App work in the gaps |
|---|---|---|---|
| Oct 19–25 | Phase 0 (Oct 19–20); Phase 1: re-pilot and test-retest (Oct 21), full labelling (Oct 22–24, unattended) | **Oct 20:** `teacher-v1` tagged. **Oct 25:** labels done | Editor: document API |
| Oct 26–Nov 1 | Phase 2: audit set, pre-registration (Oct 26–27); Phase 3: harness and CPU baselines (Oct 28–31) | **Oct 27:** `PREREGISTRATION.md` committed. **Oct 31:** `val` table for the teacher and CPU baselines | Editor: cover-letter twin |
| Nov 2–8 | Phase 4: Kaggle smoke test, 0.5B runs (Student-T and Student-D, 3 seeds), first 1.5B runs (Kaggle week 1) | | Editor: CV twin |
| Nov 9–15 | Phase 4: remaining 1.5B runs (Kaggle week 2); Phase 5: zero-shot baselines and efficiency | **Nov 15:** every `val` table complete; checkpoints, prompts and thresholds frozen | |
| Nov 16–22 | Phase 6: test labels opened once (Nov 18), final tables with CIs, failure analysis | **Nov 22:** results frozen with the commit hash | |
| Nov 23–29 | Phase 7: report and slides draft; README checked from a clean clone | **Nov 29:** full draft | |
| Nov 30–Dec 6 | Buffer: reruns if something broke, polish, presentation rehearsal | **Dec 3:** early-submission target | |
| Dec 7–10 | Only if the buffer was needed | **Dec 10:** deadline | |

The two GPU weeks hold the ~15–20 GPU-h of core runs within Kaggle's ~30 GPU-h per week, with
room for one repeat. If labelling or a GPU week slips, the buffer week absorbs it before the
early-submission target moves.

**Critical path:** Phase 0 → Phase 1 labelling → Phase 4 training → Phase 6. Phases 2, 3 and 5
run in parallel while labelling and training run on their own. Start the report skeleton in
Phase 2.

### 13.0 Work order: distillation first, app features in the waiting time

The course project is graded and has a deadline (plus an early-submission bonus); the app features
don't. Labelling and training also run unattended for long stretches. So the distillation's
critical path goes first, and app work fills the waiting time instead of competing with it.

| When | Distillation | App work |
|---|---|---|
| Now | Phase 0, then start Phase 1 (re-pilot, test-retest, full labelling: about a day unattended) | Commit the editor's segments refactor (done, tested) |
| While labelling and training run | Phases 2, 3 and 5 in the gaps | The document editor: document API, cover-letter twin, CV twin |
| After the course results (or once evaluation is stable) | Report polish, optional stretch ablations | Login and database, CV templates, job search matched to the CV |

**Freeze rule.** From the `teacher-v1` tag until labelling is done, app work stays out of what
the teacher runs:
- `scoring_agent.py`, the skill graph
- CV extraction (`cv_agent.py`, `cv_extraction_checks.py`) and job extraction (`job_agent.py`)
- the LLM routing settings

A fix that can't wait gets a new tag (`teacher-v2`) and the affected labels are regenerated; labels
from two versions are never mixed. The editor, templates, login and job search don't touch any of
these.

**Why the larger features wait:**
- **Login and database are the largest piece.** They mean accounts, storing CVs (personal data:
  consent, deletion, retention), and moving sessions out of the 24-hour checkpoint store. They
  need their own design pass.
- **CV templates come after the editor.** The editor draws an on-screen twin of the template, so
  each new template needs its twin; building the editor generically first keeps that cheap.
- **Job search depends on a jobs source.** That's an API, or scraping with its terms-of-service
  and legal questions, so it needs scoping first.

### Phase 0: repo hygiene and teacher freeze (0.5-1 day). Status: in progress
- [x] Merge the 2026-10-17 fix PRs into `develop` (#62–#68 and #69–#74).
- [ ] Merge `develop` into `feature/distillation` again (the local merge `05900e3` stops at #68),
  then push. Use a merge, not a rebase: the branch is already on GitHub.
- [ ] Restore the parked work (`git stash pop` on `feature/distillation`) and commit it in small
  commits:
  - the dataset script moved to `backend/ml/dataprep/` (not `datasets/`, which would shadow the
    Hugging Face `datasets` library)
  - the scheduler fix
  - pinned `requirements-ml.txt`
  - the extractor reading the app's prompt from `cv_agent.py` and using `chat_model("extract")`
    (no app change needed)
- [x] `resume_text_extractor.py`: run `clean_extracted_cv(result, text)` like the app does since
  #71, and test it against the app's extraction.
- [x] Compare candidate teacher models: extraction (`compare_extraction_models.py`) and the
  score's judge (`compare_judge_models.py`); results under Decisions needed, 4.
- [ ] Start `backend/ml/README.md`; run the CPU smoke test (`python -m ml.training.train_lora --smoke`).
- [ ] Decide the teacher models (Decisions needed, 4) and write them into `teacher_meta.json`.
- [ ] Tag `teacher-v1` on `develop`.

**Done when:**
- the smoke test passes on the updated branch
- the tag exists
- `git status --ignored backend/ml` shows no ignored source files

### Phase 1: teacher labels (1.5-2 days, mostly unattended). Status: not started
- [ ] `caches.py`, retry/backoff, cost from tokens, flags, `teacher_meta.json`.
- [ ] Re-pilot the 200 rows on `teacher-v1` and compare with v0.
- [ ] Test-retest on 100 rows with caches off.
- [ ] Full labelling: train (deduplicated) and test texts.

**Done when:**
- at least 95% of pairs are clean
- a summary records cost, latency, flag rates, level distribution and ρ against the train labels
  (train only)
- the re-pilot numbers are copied into Appendix A

### Phase 2: audit set and pre-registration (1 day). Status: not started
- [ ] `splits.py` (job-grouped `val`) and `audit_sample.py`.
- [ ] Label the 80 audit pairs blind with a written rubric. Add the second annotator's 30 if
  available.
- [ ] Write and commit `PREREGISTRATION.md`.
- [ ] Report skeleton: the disclosure table and Background / Data sections.

**Done when:** `PREREGISTRATION.md` is committed before any test-set evaluation.

### Phase 3: evaluation harness and CPU baselines (1.5-2 days). Status: not started
- [ ] `metrics.py`: clustered bootstrap, fitted thresholds, QWK. Unit tests against hand-computed
  cases.
- [ ] `eval_fidelity.py`, `eval_external.py`.
- [ ] `baselines.py`: embedding cosine, TF-IDF ridge / LR, resume prior, majority.

**Done when:** one command prints the `val` table for the teacher and all CPU baselines.

### Phase 4: students (2-3 days, GPU, unattended). Status: not started
- [ ] Smoke test on Kaggle with the real labels (200 examples, 1 epoch).
- [ ] Student-T and Student-D at 0.5B × 3 seeds, and at 1.5B × 1-2 seeds.
- [ ] `run_inference.py` on `val`. Choose checkpoints.

**Done when:** the `val` fidelity and validity tables include all students.

### Phase 5: zero-shot baselines and efficiency (1 day). Status: not started
- [ ] Zero-shot Qwen 0.5B / 1.5B and `gpt-4o-mini` on `val`.
- [ ] `eval_latency_cost.py`: teacher (cold, cache off), API zero-shot, students on GPU, 0.5B on
  CPU.

**Done when:** the efficiency table and break-even volume are computed.

### Phase 6: final evaluation (1-1.5 days). Status: not started
- [ ] Freeze checkpoints, prompts and thresholds. Open the test labels once.
- [ ] Final tables with clustered CIs, the pre-registered comparisons, and the audit-set results.
- [ ] `failure_analysis.py` and the qualitative categories.

**Done when:** results are written to `backend/data/ml/results/` with the commit hash, and the
numeric predictions are exported for release.

### Phase 7: report, presentation, reproducibility package (2 days). Status: not started
- [ ] Report: disclosure table on page 1, RQs, method, results with CIs, failure analysis,
  threats, limitations.
- [ ] Slides.
- [ ] Verify the README from a clean clone. Link the Kaggle notebook.

**Done when:** a clean clone reproduces the tables from the released predictions.

**Total core path:** roughly 10-13 working days, of which about 3 are unattended labelling and
training. An early submission is realistic once the deadline is known.

### Decisions needed

1. ~~The deadline.~~ Decided: 2026-12-10, aiming for an early submission on 2026-12-03 (see the
   schedule). Still open: the presentation date, and how the early-submission bonus is counted.
2. A second annotator for 30 audit pairs. Optional, but it makes the label-noise estimate much
   stronger.
3. Whether the approved proposal words the RQs or the metric split differently from Section 3. If
   it does, Section 3 adopts that wording.
4. **The teacher's models, before the tag.** Compared on 2026-10-17; raw results in
   `data/ml/model_comparison/` and `data/ml/judge_comparison/` (git-ignored), scripts committed.
   Only dated or versioned model ids are used, so the teacher can't change under a moving alias.
   - **Extraction: decided, `mistralai/mistral-large-4-0`.** 12 train resumes, 2 runs each, 7
     models reachable from China through OpenRouter, plus OpenAI's models through a direct key:

     | model | failed | content kept | lines reworded | same bullet placement | per 1,000 |
     |---|---|---|---|---|---|
     | mistralai/mistral-large-4-0 | 0 | 0.862 | 1% | 0.974 | $6.21 |
     | minimax/minimax-m3 | 0 | 0.851 | 0.2% | 0.966 | $2.44 |
     | deepseek/deepseek-v4.1-flash-1017 | 0 | 0.826 | 0.3% | 0.926 | $1.28 |
     | deepseek/deepseek-v4-pro-0813 | 1 | 0.813 | 2% | 0.961 | $4.78 |
     | qwen/qwen3.8-flash | 0 | 0.772 | 0% | 0.849 | $1.09 |
     | openai/gpt-6.1-sol (direct) | 0 | 0.662 | 43% | 0.654 | $23.06 |
     | openai/gpt-6-luna (direct) | 0 | 0.625 | 50% | 0.484 | $1.22 |

     No model returns byte-identical extractions twice, so extractions are cached per distinct
     text (`caches.py`) and the labels are generated once. OpenAI's models reword about half the
     lines and drop content (they also only accept temperature 1); `qwen3.8-max-0902` refuses
     requests without reasoning.
   - **The score's judge (`LLM_MODEL_JUDGE`): open, deepseek or mistral.** 20 train pairs
     extracted once with Mistral, each scored twice per judge:

     | judge | unjudged | mean / max change of the score between runs | same level | same skill verdict | full labelling |
     |---|---|---|---|---|---|
     | mistralai/mistral-large-4-0 | 0% | 0.010 / 0.029 | 95% | 93.4% | $27 |
     | deepseek/deepseek-v4.1-flash-1017 | 0% | 0.018 / 0.074 | 95% | 90.5% | $6 |
     | minimax/minimax-m3 | 97% | (its batched call failed) | | | |

     Mistral halves the run-to-run noise of the teacher's score for about $21 more.
   - **Either way:** the labels depend on this choice, it is recorded in `teacher_meta.json`, and
     it can't change during labelling.

### How the plan maps to the grading

| Criterion | Where it is earned |
|---|---|
| Technical soundness | Grouped splits, test-retest ceiling, resume-prior baseline, audit set, clustered CIs, pre-registration |
| Technical excitement | Fidelity vs. validity of a distilled hybrid pipeline; the RQ3 label-source ablation; per-dimension transfer analysis |
| Clarity | One primary metric per RQ; disclosure table on page 1 |
| Individual contribution | Everything under `backend/ml/`, with commit ranges |
| Reproducibility bonus | Pinned requirements, README, seeds, released predictions, teacher metadata |
| Ethics | Prior-work disclosure, AI-assistance disclosure, no text redistribution |

---

## Appendix A: data findings and pilot v0 (2026-09-17 scorer, outdated teacher)

### A.1 HF dataset (`cnamuangtoun/resume-job-description-fit`, report in `data/ml/hf_eval_cache/dataset_report.json`)

| | train | test |
|---|---|---|
| rows | 6241 | 1759 |
| No Fit / Potential Fit / Good Fit | 3143 / 1556 / 1542 | 857 / 444 / 458 |
| unique resumes / jobs | 642 / 280 | 477 / 71 |
| empty texts, duplicate pairs | 0, 7 | 0, 0 |
| resume tokens p50 / p90 / max (Qwen2.5) | 1014 / 1846 / 4256 | 1019 / 1843 / 4057 |
| job tokens p50 / p90 / max | 430 / 919 / 1432 | 421 / 1061 / 1563 |

- 476 of 477 test resumes appear in train. Jobs and pairs are disjoint.
- About 10 rows per resume and ~22 per job.
- ~50% No Fit, so macro-F1 rather than accuracy for classification.
- No card, no licence, no description (Hugging Face API, 2026-10-17).

### A.2 Pilot v0 (200 seeded train rows, `data/ml/teacher_labels/pilot.summary.json`)

- **Completion:** 200/200 ok; 0 degenerate CVs, 0 errors, 0 flagged rows.
- **Cost:** $0.34 ($0.0017 per row; ~$10.50 extrapolated to 6,241 rows before extraction
  caching).
- **Latency:** p50 44 s / p95 100 s per row with 8 workers (~30 min wall-clock).
- **Levels:** 146 low / 53 medium / 1 high / 0 excellent.
- **Aggregate score:** 0.135-0.653, mean 0.38.
- **Dimension means:** hard_skills 0.32, experience 0.48, seniority 0.62, domain 0.40,
  ats_keywords 0.18.
- **Experience defect (now fixed in the app):** 158/200 rows fell back to the constant "Could not
  estimate candidate years" (0.4), because `_estimate_candidate_years` ignored
  `ExperienceEntry.duration`. It now reads `duration` and parses more date formats
  (`fix/experience-date-ranges`). This is one of the reasons the v0 numbers are outdated.
- **Agreement with train labels (computed 2026-10-17):**
  - Spearman 0.16 overall.
  - Mean aggregate by label: No Fit 0.36 / Potential Fit 0.42 / Good Fit 0.39.
  - Per dimension: domain 0.26, hard_skills 0.20, experience 0.05, seniority −0.06,
    ats_keywords −0.04.
  - The 200 rows cover 162 resumes and 120 jobs.

### A.3 Resume-prior diagnostic (train only, computed 2026-10-17)

- **Setup:** 20% of train jobs held out (1,282 rows; 99% of their resumes seen in the rest of
  train).
- **Resume prior:** ρ 0.28, macro-F1 0.29, accuracy 0.34.
- **Majority class:** accuracy 0.45.
- **Label consistency:** 19% of resumes and 13% of jobs carry a single label across all their
  rows.

## Appendix B: related work pointers

- **ConFit v3** and **arXiv 2505.17612** (agent distillation) both use plain SFT as the base
  recipe. Cite them in Related Work and discuss their extensions (listwise re-ranking,
  retrieval / code-tool distillation). Implementing them in full is out of scope.
- A listwise ranking loss in place of pointwise JSON regression is a bonus ablation, not a
  prerequisite.
- A custom regression head on the base model is deliberately not built: it is a bigger surface
  and harder to reproduce. It belongs in future work.
