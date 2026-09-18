# NLP Course Project: Distilling Compatibility Scoring into a Small Fine-Tuned Model

Status: draft implementation plan, 2026-09-17.

## 0. Scope & attribution (read this before writing anything else)

This document plans **new work only**. It does not claim any part of the existing
JobWriterAI application — `backend/app/agents/scoring_agent.py`, its skill-matching
cascade, `backend/app/data/skill_graph.json`, or the LangGraph orchestrator — as a
contribution of this course project. That code predates the course and is treated
here strictly as:

1. **The teacher** whose behavior we distill (a black box we call, not code we wrote
   for this project), and
2. **Reused infrastructure** (its Pydantic schemas, its text-extraction tools) that
   the *data preparation* scripts call into, the same way a project might call any
   third-party library.

Everything under the new `backend/ml/` directory (Section 1) is the actual course
contribution: teacher-label generation harness, external-dataset prep, LoRA/QLoRA
training, and the evaluation protocol. Keep this line intact in the report: describe
`scoring_agent.py` in a "Background / Existing System" subsection, not in
"Methodology" or "Contributions."

## 1. Repo layout for the new work

New, isolated top-level module — nothing added to `backend/app/agents/`:

```
backend/ml/
  __init__.py
  config.py                        # paths, seeds, model registry, constants
  requirements-ml.txt              # torch/transformers/peft/bitsandbytes/datasets/trl — NOT in backend/requirements.txt

  labeling/
    __init__.py
    resume_text_extractor.py       # NEW: raw text -> ResumeInfo (fills the gap cv_agent.py doesn't cover)
    generate_teacher_labels.py     # main script: text pairs -> CompatibilityReport -> JSONL
    embedding_cache.py             # disk-persisted cache keyed by (skill_text, model) to cut OpenAI cost

  data/
    __init__.py
    prepare_hf_eval_dataset.py     # download cnamuangtoun/resume-job-description-fit, split handling
    build_training_examples.py     # teacher labels (JSONL) -> instruction/completion training format
    schema.py                      # dataclasses/TypedDicts for on-disk record formats

  training/
    __init__.py
    train_lora.py                  # LoRA/QLoRA SFT script (Kaggle notebook entry point)
    model_registry.py              # base model choices + LoRA configs per model

  evaluation/
    __init__.py
    run_inference.py               # shared inference wrapper for: distilled model, zero-shot baseline, teacher pipeline
    eval_agreement.py              # agreement-with-teacher + agreement-with-external metrics
    eval_latency_cost.py           # latency/cost benchmarking harness
    failure_analysis.py            # surfaces largest-divergence examples for qualitative writeup

  inference/
    __init__.py
    distilled_scorer.py            # STRETCH ONLY: thin wrapper to call the fine-tuned model like a tool

  notebooks/
    train_kaggle.ipynb             # thin wrapper that calls training/train_lora.py on Kaggle GPU

backend/data/ml/                   # git-ignored already (backend/data/ is in .gitignore) — all generated
  teacher_labels/                  #   artifacts land here: labels, caches, checkpoints, HF dataset cache
  hf_eval_cache/
  embedding_cache.sqlite
  checkpoints/
```

Why a new top-level `backend/ml/` instead of `backend/app/ml/`: it's not part of the
FastAPI app (`backend/app/`) at all until the stretch-goal integration step, and
keeping it a sibling makes the "new work vs. existing app" boundary visually obvious
in the repo tree — useful when your professor looks at the diff.

Add to `.gitignore` (root, next to the existing `backend/data/` line — already
covers `backend/data/ml/`, so no change needed there). Do add:
```
backend/ml/notebooks/*.ipynb_checkpoints/
```

## 2. Teacher-label generation

### 2.1 The gap: `cv_agent.py` has no raw-text extraction path

`extract_resume_info` (`backend/app/agents/cv_agent.py:43`) only accepts
`pdf_path`/`pdf_bytes` — it opens a PDF with `fitz`, extracts text, then runs a
structured-output LLM call. The HF dataset's `resume_text` column is already plain
text, so there's no PDF to open. `job_agent.retrieve_text` (`job_agent.py:67`)
already does exactly the text→schema step we need for jobs, so nothing new is
needed on that side.

Write `backend/ml/labeling/resume_text_extractor.py` as a **data-prep utility**
that mirrors `extract_resume_info`'s LLM step (same `ResumeInfo` schema, same
system-prompt strategy) but skips PDF parsing:

```python
# backend/ml/labeling/resume_text_extractor.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import ResumeInfo  # reused, not duplicated

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract all information from the resume text. ..."),  # same prompt text as cv_agent.py
    ("user", "{text}"),
])

def extract_resume_info_from_text(text: str) -> ResumeInfo:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = _PROMPT | llm.with_structured_output(ResumeInfo)
    return chain.invoke({"text": text})
```

Note this is infrastructure to *feed* the teacher pipeline, not a modification of
`cv_agent.py` and not part of the trained model.

### 2.2 Where the (resume, job) pairs come from

Use the **HF dataset's `train` split** (6.24k rows) purely as a source of raw
`(resume_text, job_description_text)` pairs — **discard its `label` column
entirely for this purpose**. The dataset's own labels are reserved exclusively for
the independent evaluation in Section 5.2; the `train` split's text is only ever
used as unlabeled input to generate *your own* teacher labels. Keep this
distinction explicit in the report — it's what makes the HF `test` split a valid
independent anchor rather than leaked signal.

(Columns verified 2026-09-17: `resume_text` / `job_description_text` / `label`.)

### 2.3 Label generation script

```python
# backend/ml/labeling/generate_teacher_labels.py (skeleton)
"""
For each (resume_text, job_description_text) row from the HF train split:
  1. job_data = job_agent.retrieve_text.invoke({"info": job_description_text})
  2. cv_data  = resume_text_extractor.extract_resume_info_from_text(resume_text)
  3. skip row if extraction is degenerate (empty skills AND empty experience) —
     real Kaggle resume dumps are noisy; log a drop-rate stat for the report
  4. report = calculate_compatibility_score_v2.invoke({"cv_json": ..., "job_json": ...})
  5. append {row_id, resume_text, job_description_text, cv_data, job_data, report}
     to a resumable JSONL file (one line per row, flush per row)
"""
```

Requirements this script must satisfy (none of these require touching
`scoring_agent.py`):

- **Resumable / idempotent**: keyed by row id, skip rows already in the output
  JSONL. A multi-hour run against a paid API will get interrupted; don't restart
  from zero.
- **Persisted embedding cache**: `scoring_agent.py`'s `_get_single_embedding_cached`
  is `lru_cache`-only (in-process, lost on restart). Skill names recur heavily
  across thousands of job postings ("python", "react", "aws" appear constantly),
  so wrap `get_embeddings`/`get_skill_embedding` calls behind a disk cache
  (`backend/ml/labeling/embedding_cache.py`, e.g. a sqlite file at
  `backend/data/ml/embedding_cache.sqlite` keyed by `(text, model)`) that survives
  across runs. This is the single biggest cost lever.
- **Concurrency + backoff**: process rows with a small thread/async pool (e.g.
  8-16 concurrent) with exponential backoff on rate-limit errors — the pipeline's
  own LLM/embedding calls are per-row synchronous.
- **Pilot before scale**: run on ~150-250 rows first, inspect a sample of the
  resulting `CompatibilityReport`s for sanity (do scores look reasonable? is the
  degenerate-extraction drop rate acceptable?), and extrapolate cost/time before
  committing to the full ~6k run. Log $ spent (token counts × published pricing)
  and wall-clock time in the pilot so the report's "Data" section has real numbers.

### 2.4 Pilot results (2026-09-17, 200 seeded-random train rows, `data/ml/teacher_labels/pilot.summary.json`)

- 200/200 ok; 0 degenerate resumes, 0 errors, 0 silent LLM/embedding failures.
- LLM cost $0.34 ($0.0017/row, ~$10.50 extrapolated to 6241 rows, before extraction caching).
- Latency p50 44 s / p95 100 s per row with 8 workers (~30 min wall-clock).
- **Teacher output is compressed**: levels 146 low / 53 medium / 1 high / 0 excellent;
  aggregate 0.135–0.653 (mean 0.38). Per dimension (mean): hard_skills 0.32,
  experience 0.48, seniority 0.62, domain 0.40, ats_keywords 0.18.
- **Teacher defect, experience dimension**: 158/200 rows fall back to the constant
  "Could not estimate candidate years" (0.4). `scoring_agent._estimate_candidate_years`
  never reads `ExperienceEntry.duration`, where 153 of those 158 rows have their dates,
  and `_DATE_RANGE_RE` wouldn't match the `MM/YYYY-MM/YYYY` format anyway.
  **Open decision:** fix it in the existing app before full labeling (a pre-existing-app
  bug fix, its own commit, not a course contribution) or distill the teacher as-is and
  document it as a limitation.

## 3. Data prep for the HF external eval dataset

`backend/ml/data/prepare_hf_eval_dataset.py`:

- `datasets.load_dataset("cnamuangtoun/resume-job-description-fit")`, cache under
  `backend/data/ml/hf_eval_cache/`.
- Confirm and record: exact column names, class balance of `label` (3-class:
  expect something like `No Fit` / `Potential Fit` / `Good Fit`), 6.24k/1.76k
  split sizes.
- Preprocessing: strip boilerplate/HTML artifacts if present, drop empty-text
  rows, record a length histogram (resume/job text token counts) — this
  determines your model's max sequence length and truncation strategy in
  training (Section 4).
- **Never touch `test` split text or labels until final evaluation.** `train`
  split text feeds Section 2; `test` split is untouched until Section 5.2.

### 3.1 Verified findings (2026-09-17, `data/ml/hf_eval_cache/dataset_report.json`)

| | train | test |
|---|---|---|
| rows | 6241 | 1759 |
| No Fit / Potential Fit / Good Fit | 3143 / 1556 / 1542 | 857 / 444 / 458 |
| unique resumes / jobs | 642 / 280 | 477 / 71 |
| empty texts, duplicate pairs | 0, 7 | 0, 0 |
| resume tokens p50 / p90 / max (Qwen2.5) | 1014 / 1846 / 4256 | 1019 / 1843 / 4057 |
| job tokens p50 / p90 / max | 430 / 919 / 1432 | 421 / 1061 / 1563 |

- **Train/test share resumes: 476 of 477 test resumes appear in train.** Jobs and
  pairs are fully disjoint. The HF `test` split therefore measures generalization
  to **unseen job descriptions**, not unseen candidates. State this in the report's
  evaluation protocol; don't claim resume-level generalization.
- **Few unique texts.** ~10 rows per resume, ~22 per job. Consequences:
  (1) the teacher-validation split (Section 5.1) must be **grouped by job**, or
  agreement is inflated by memorized jobs (`train_lora.py` does this);
  (2) caching resume/job extraction by text cuts labeling LLM calls ~10-20×.
- **Class imbalance**: ~50% `No Fit` — report macro-F1, not just accuracy.
- No cleaning needed (no empties); lengths set truncation to ~p90
  (resume 1536 / job 768 tokens) in `train_lora.py`.

## 4. Training script skeleton

### 4.1 Design decision: raw text in, structured JSON out

The distilled model takes **raw `(resume_text, job_description_text)` text** as
input — not the structured `cv_data`/`job_data` JSON the existing pipeline
consumes internally. Reasons:

- It matches the external HF eval set's input format exactly (raw text pairs),
  so no extraction step is needed at eval-on-external-data time.
- It makes the distilled model a genuine end-to-end fast-path replacement
  (text → score), not something that still depends on the existing app's
  extraction agents at inference time.

Output target (generative SFT, formatted as JSON in the completion):
```json
{"aggregate_score": 0.71, "level": "high",
 "dimensions": {"hard_skills": 0.68, "experience": 0.9, "seniority": 0.75, "domain": 0.6, "ats_keywords": 0.72}}
```
Include the five dimension scores — the teacher pipeline computes them for free,
and they materially strengthen the failure-analysis section (Section 6) for
close-to-zero extra training complexity. **Do not** attempt to reproduce the
structured `GapAnalysis` skill lists (matched/transferable/missing skill names)
as a core target — open-vocabulary generation with a fiddly eval story is a bad
time investment for a 0.5-3B model on a course timeline. Keep it as an explicit
stretch/discussion item ("could this scale to structured gap output?") rather
than a deliverable.

Both aggregate_score and level are included in the target JSON. level is the primary evaluation target — closed label space, matches the external dataset's granularity, and is robust to malformed float output. aggregate_score is secondary, used for correlation-based metrics and per-dimension failure analysis.") and tweak the opening of 5.1/5.2 to say "primary metric: band accuracy/macro-F1" before listing the continuous-score metrics.

### 4.2 Base model + method

- **Primary**: `Qwen2.5-1.5B-Instruct` — good quality/compute tradeoff for
  Kaggle's free GPU tier.
- **Ablation (if time allows)**: `Qwen2.5-0.5B-Instruct` — cheaper, tests how far
  distillation quality degrades at the smallest size, which is a natural
  "quantitative results" axis for the report.
- **Stretch**: `Qwen2.5-3B-Instruct` if compute budget and time allow a second
  full run.
- **Method**: QLoRA (4-bit base via `bitsandbytes` + LoRA adapters via `peft`),
  standard causal-LM SFT loss with the prompt tokens masked out of the loss
  (only the completion JSON contributes). Use `trl`'s `SFTTrainer` to avoid
  hand-rolling the masking/collation logic — this is the same simplicity the
  agent-distillation paper (arXiv 2505.17612) and ConFit v3 both use as their
  base recipe (plain SFT), before any listwise/re-ranking additions.
- Explicitly **not** building a custom regression head on the base model —
  bigger architectural surface, harder to reproduce, and generative JSON output
  is sufficient for the score/band targets. Mention this tradeoff in the
  report's limitations/future-work, don't implement it.

```python
# backend/ml/training/train_lora.py (skeleton)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype="bfloat16")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                           task_type="CAUSAL_LM")
    return get_peft_model(model, lora_cfg), tok

def format_example(row):
    prompt = f"Resume:\n{row['resume_text']}\n\nJob:\n{row['job_description_text']}\n\nCompatibility JSON:"
    completion = json.dumps({"aggregate_score": row["aggregate_score"], "level": row["level"],
                              "dimensions": row["dimensions"]})
    return {"text": prompt + " " + completion}

def main():
    model, tok = load_model()
    train_ds, val_ds = load_teacher_label_splits()  # 90/10 of backend/data/ml/teacher_labels/*.jsonl
    trainer = SFTTrainer(model=model, tokenizer=tok,
                          train_dataset=train_ds.map(format_example),
                          eval_dataset=val_ds.map(format_example),
                          args=SFTConfig(output_dir="backend/data/ml/checkpoints/qwen1_5b",
                                         per_device_train_batch_size=4,
                                         gradient_accumulation_steps=4,
                                         num_train_epochs=3, learning_rate=2e-4,
                                         max_seq_length=1024, bf16=True, save_strategy="epoch"))
    trainer.train()
```

### 4.3 Kaggle compute budget

Kaggle gives ~30 GPU-hours/week (T4×2 or P100). For ~5.5k training examples
(after the pilot-validated drop rate), 3 epochs, seq len ~1024-2048 (truncate
long resumes — check the length histogram from Section 3), batch 4 ×
grad-accum 4:

- **Qwen2.5-0.5B**: ballpark 30-60 min/epoch on a single T4 → ~1.5-3h total.
- **Qwen2.5-1.5B**: ballpark 1-2h/epoch → ~3-6h total.
- **Qwen2.5-3B (stretch)**: likely needs T4×2 or a P100, ~2-4h/epoch → budget a
  full day of Kaggle quota.

Always run a 200-example, 1-epoch smoke test first to catch OOM/formatting bugs
before burning quota on a full run — this is the single highest-leverage time
saver on Kaggle's session limits.

## 5. Evaluation script skeleton

Two independent axes, both required per the assignment; do not conflate them.

### 5.1 Agreement with teacher (held-out split of your own labels)

Hold out ~10% of the teacher-labeled set (never trained on). For each row, run
the distilled model and compare to the teacher's `CompatibilityReport`:
- MAE/RMSE + Pearson/Spearman correlation on `aggregate_score`.
- Band classification accuracy + confusion matrix on `level`.
- Per-dimension MAE (5 numbers) — feeds Section 6's failure analysis (which
  dimension does the small model struggle with most?).

### 5.2 Agreement with external labels (HF `test` split, 1.76k, untouched until now)

Label-space mismatch to handle explicitly: the teacher pipeline's `level` is a
4-band scale (`low/medium/high/excellent`); the HF dataset's `label` is 3-class
(`No Fit`/`Potential Fit`/`Good Fit`). Don't silently force one into the other —
report both:
- A documented band-collapse mapping (e.g. merge `high`+`excellent` → one class)
  for a direct accuracy/macro-F1/Cohen's-kappa comparison.
- An ordinal correlation (Spearman) between the model's continuous
  `aggregate_score` and an ordinal encoding of the 3-class label
  (`No Fit=0, Potential=1, Good=2`) — this sidesteps the mapping question
  entirely and is arguably the cleaner generalization metric to lead with.

### 5.3 Baselines run through the identical harness

- **(a) Existing cascade pipeline itself**, run directly on a cost-bounded sample
  (e.g. 300-500 rows, not all 1.76k — real paid API calls) of the HF `test` set.
  This is your upper-bound reference: how well does the *teacher* agree with the
  external ground truth? The distilled model's job is to approach this number
  cheaply, not necessarily beat it.
- **(b) Zero-shot prompting** of a general LLM (reuse `gpt-4o-mini`, and/or the
  same base Qwen model with no fine-tuning) on the identical prompt format from
  Section 4.1, on both eval sets.
- **(c) ConFit v3 / arXiv 2505.17612 comparison**: treat as related-work citations
  and a discussion of how their listwise re-ranking / retrieval+code-tool
  distillation ideas relate to your simpler SFT approach — implementing either
  in full is disproportionate to a course timeline. If time remains after
  Section 4-6 are done, a lightweight ablation (e.g. listwise ranking loss
  instead of pointwise regression-via-JSON) is a good bonus, not a prerequisite.

### 5.4 Latency/cost harness

`backend/ml/evaluation/eval_latency_cost.py`: same N inputs through all three
systems (distilled model, existing cascade, zero-shot baseline), on the same
machine, warm-started, averaged over multiple runs:
- Wall-clock latency per call (report median + p95, not just mean).
- $ cost per call: token-counted × published pricing for the cascade/zero-shot
  (real OpenAI calls); compute-only for the distilled model (report both a
  Kaggle-GPU number and, if feasible, a CPU-inference number since that's the
  more realistic "cheap fast-path" deployment target).

### 5.5 Failure analysis

`backend/ml/evaluation/failure_analysis.py`: rank held-out + external examples
by `|predicted - teacher|` (and separately by disagreement with external label),
dump the top ~15-20 divergent cases with resume/job excerpts for manual
qualitative categorization (e.g. "model over-weights keyword overlap and misses
seniority mismatch", "model saturates scores toward the middle band"). This
qualitative section is explicitly graded — don't skip it or leave it to the
night before submission.

## 6. Does the current implementation need further process for evaluation?

Yes, but only in the new harness, never in `scoring_agent.py` itself:

- **Persisted embedding cache** (Section 2.3) — the existing `lru_cache` doesn't
  survive process restarts and doesn't help across the teacher-labeling script's
  thousands of repeated skill names.
- **Concurrency + retry/backoff wrapper** around the per-row synchronous
  `calculate_compatibility_score_v2` calls — the pipeline itself has no batching
  across rows (only within one job's unresolved-skill LLM call), so the label
  generation script needs its own thread/async pool.
- **Cost/time pilot-then-scale discipline** (Section 2.3) — running the full
  ~6k-row teacher-labeling pass blind, without a pilot, risks burning API budget
  on a bug or on a bad extraction-quality rate.
- **Determinism caveat for the report**: `temperature=0` calls to OpenAI are
  *not* perfectly deterministic across API-side model updates; note the exact
  model snapshot used (e.g. `gpt-4o-mini`) and the date labels were generated, so
  the "teacher" is reproducibly described even if not byte-for-byte
  reproducible.

No other changes to the existing scoring pipeline are needed — it already
produces exactly the structured, multi-dimensional signal the training/eval
scripts need to consume.

## 7. Ordered implementation plan with time estimates

Estimates assume solo, part-time (a few hours/day) work; compress if you can
block out full days.

| # | Phase | Key deliverable | Est. time |
|---|-------|------------------|-----------|
| 0 | Scaffolding | `backend/ml/` tree, `requirements-ml.txt`, `config.py`, confirm professor sign-off | 0.5 day |
| 1 | HF dataset prep | `prepare_hf_eval_dataset.py` run, column/schema confirmed, length histogram | 0.5–1 day |
| 2 | Teacher labeling | `resume_text_extractor.py` + `generate_teacher_labels.py`, pilot run (~200 rows), then full ~5.5k run w/ cache+retry | 1.5–2 days |
| 3 | Training data format | `build_training_examples.py`, train/val split, dataset schema doc | 0.5–1 day |
| 4 | Training | Kaggle notebook wiring, smoke test, full LoRA run(s) for 1.5B (+0.5B ablation if time) | 2–3 days |
| 5 | Evaluation harness | `eval_agreement.py`, `eval_latency_cost.py`, run against both baselines + both eval axes | 1.5–2 days |
| 6 | Failure analysis | `failure_analysis.py`, manual categorization write-up | 1 day |
| 7 | Report + poster | Incorporate citations, results tables/figures, scope section from Section 0 | 1–1.5 days |
| — | Stretch: app integration | `backend/ml/inference/distilled_scorer.py`, optional fast-scoring toggle in orchestrator | 1–2 days, cut first under time pressure |

Core path (0-7): roughly **9-12 working days**. Start Phase 7's report skeleton
early (right after Phase 2) and fill in results incrementally as Phases 4-6
land — don't treat it as a final sequential step, since the early-bird bonus
rewards finishing sooner, not finishing everything-then-writing.
