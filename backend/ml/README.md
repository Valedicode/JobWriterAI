# Distilling the compatibility scorer (NLP course project)

Everything under `backend/ml/` is the course project. The app it distils (`backend/app/`) is prior
work and is treated as the teacher, a black box; see `NLP_PROJECT_PLAN.md` §1.

Generated files go to `backend/data/ml/` (git-ignored), or to `$ML_ARTIFACTS` when that is set.

## Setup

From `backend/`, in the `jobwriterai` conda env:

```
pip install -r requirements.txt -r ml/requirements-ml.txt
python -m pytest ml/tests
```

`backend/.env` needs two keys:

- `OPENROUTER_API_KEY` for the teacher's chat models.
- `OPENAI_API_KEY` for its embeddings (`text-embedding-3-small`). OpenRouter refuses OpenAI's models
  from some regions, so the app calls OpenAI directly when this key is set.

## The teacher's settings

The teacher is the app at the tag `teacher-v1`, with these settings (in `.env` or the shell):

```
LLM_MODEL_EXTRACT=deepseek/deepseek-v4.1-flash-1017
LLM_MODEL_JUDGE=deepseek/deepseek-v4.1-flash-1017
LLM_PROVIDER=<provider slug>   # one OpenRouter host, see NLP_PROJECT_PLAN.md, Decisions
```

`LLM_PROVIDER` pins every chat call to one host with no fallback, so the labels can't come from a
mix of hosts and quantisations. DeepSeek's own endpoint is not an option for an OpenRouter account
that excludes providers which train on prompts; OpenRouter then returns 404 "No endpoints found".

The labelling script refuses to run unless the teacher is frozen: HEAD on a `teacher-*` tag, no
uncommitted changes in `app/` or `ml/`, `LLM_PROVIDER` set, and versioned model ids. `--allow-unfrozen`
runs a trial anyway and marks its metadata `"frozen": false`.

## Labelling

Run the labels from a checkout of the tag, so later app work can't change the teacher mid-run.
From the repo root:

```
git worktree add ../jobwriterai-teacher-v1 teacher-v1
cp backend/.env ../jobwriterai-teacher-v1/backend/.env
cd ../jobwriterai-teacher-v1/backend
export ML_ARTIFACTS=<repo root>/backend/data/ml   # write into the main checkout's artifacts
```

Then:

```
# Re-pilot: the v0 pilot's 200 seeded train rows
python -m ml.labeling.generate_teacher_labels --limit 200 --workers 16 --out repilot.jsonl

# Test-retest: 100 rows twice, with fresh extractions and embeddings each time
python -m ml.labeling.generate_teacher_labels --limit 100 --no-cache --out retest-a.jsonl
python -m ml.labeling.generate_teacher_labels --limit 100 --no-cache --out retest-b.jsonl
python -m ml.labeling.teacher_retest   # agreement of the two runs -> teacher_labels/retest.summary.json

# Full runs: every train row (repeated pairs dropped) and every test row
python -m ml.labeling.generate_teacher_labels --limit 0 --workers 16 --out train.jsonl
python -m ml.labeling.generate_teacher_labels --split test --limit 0 --workers 16 --out test.jsonl
```

Each run writes, under `teacher_labels/`:

| File | Content |
|---|---|
| `<out>.jsonl` | one record per scored row: status, the teacher's report, flags, attempts, tokens, spend |
| `<out>.extractions.jsonl` | each text extracted in the run, with its result |
| `<out>.teacher_meta.json` | tag, commit, models, host, prices, packages, dataset revision, dates |
| `<out>.summary.json` | statuses, flags, spend, latency, levels, scores; for train also the agreement with the dataset labels |

The retest summary gives, per score (aggregate and each dimension), Spearman's ρ, ICC(A,1) and the
absolute differences between the two runs, plus how often the level and the extractions came out the
same. It is the students' noise ceiling: agreeing with the teacher beyond it only fits the teacher's
noise.

The test split's labels are never read. Train labels are read only for the summary.

**Resuming.** Rerunning the same command continues the file: finished rows are skipped and failed
rows are tried again. A run continuing a file must use the same teacher; anything else is refused.

**Stopping.** A row whose score was reached despite a failed API call is retried (the scorer would
otherwise return a wrong score without saying so). A fatal error (bad key, no credit, a host or
region refused, no endpoint) or 25 failures in a row stops the run with exit code 2; finished rows
are kept.

**Caches.** Extractions and embeddings are kept in `teacher_cache/` across runs. Each distinct text
is extracted once per teacher version; `--no-cache` uses fresh ones for the run.

## Split and audit set

```
python -m ml.dataprep.splits         # job-grouped train/val -> splits/splits.json (34 val jobs, 721 rows)
python -m ml.dataprep.audit_sample   # 80 blind val pairs -> audit/audit_sheet.csv and audit_pairs.md
```

Label the audit pairs in `audit_sheet.csv` following `ml/AUDIT_RUBRIC.md`, reading the texts in
`audit_pairs.md`. Both files contain resume text, so they stay in the git-ignored `data/ml/audit/`.
`ml/PREREGISTRATION.md` fixes the hypotheses, metrics and comparisons before any model is trained.

## Training the students

Export the targets locally, after the labels and the split:

```
python -m ml.training.targets   # -> students/targets.jsonl: row id, split, both targets; no text
```

`targets.jsonl` holds no resume or job text; the training script loads the texts from the HF
dataset (at the pinned revision) by row id. The training code doesn't import the app, and the repo
is public, so Kaggle clones the code and needs only `targets.jsonl` uploaded.

On Kaggle:

1. Create → New Dataset: upload `backend/data/ml/students/targets.jsonl`, name it
   `jobwriterai-targets`, and keep it **private** (it holds the teacher's outputs and the dataset's
   labels, which aren't ours to redistribute).
2. New notebook: Settings → Accelerator **GPU T4**, Internet **on**; Add Input →
   `jobwriterai-targets`. Then:

   ```
   !git clone -q -b develop https://github.com/Valedicode/JobWriterAI.git /kaggle/working/repo
   %cd /kaggle/working/repo/backend
   !git rev-parse --short HEAD
   !pip install -q transformers==5.17.0 peft==0.21.0 trl==1.13.0 accelerate==1.15.0 datasets==5.0.1 bitsandbytes
   %env ML_ARTIFACTS=/kaggle/working/artifacts
   !python -m ml.training.train_lora --target teacher --model Qwen/Qwen2.5-0.5B-Instruct --seed 42 \
       --targets /kaggle/input/jobwriterai-targets/targets.jsonl --limit 200 --epochs 1   # smoke first
   ```

   Record the commit the clone prints and train every run from it (`!git checkout <commit>` after
   the clone in later sessions), so all students share the same training code. If the targets
   aren't found, `!find /kaggle/input -name targets.jsonl` shows where Kaggle put them.

   Keep Kaggle's own torch (don't install `requirements-ml.txt` whole); record the `bitsandbytes`
   version from `pip freeze` and pin it in `requirements-ml.txt` after the first run.
3. Then the real runs, dropping `--limit` and `--epochs` (3 epochs; the best `val` epoch is kept):
   `--target teacher|dataset`, `--model Qwen/Qwen2.5-0.5B-Instruct` with `--seed 42`, `43`, `44`,
   and `Qwen/Qwen2.5-1.5B-Instruct` with `--seed 42` (and `43` if the GPU quota allows).

Each run writes `checkpoints/student-<t|d>-<model>-seed<seed>/`, with the kept adapter in
`adapter/`. Download the adapters and the `trainer_state.json` files into `backend/data/ml/checkpoints/`.

## Evaluation

Every system writes a prediction file, `predictions/<system>.json`, with a score per `val` and test
row. The CPU baselines (teacher, TF-IDF + ridge, TF-IDF + logistic regression, resume prior,
majority class, and with `--embeddings` the embedding cosine, which calls OpenAI for a few cents):

```
python -m ml.evaluation.baselines --embeddings
python -m ml.evaluation.evaluate          # the val table -> results/val.json
```

The table gives, per system, fidelity (Spearman ρ with the teacher's aggregate) and validity
(Spearman ρ with the dataset label, then macro-F1, QWK and accuracy after two thresholds fitted on
`val`), with 95% CIs from a bootstrap that resamples whole jobs. Once the students' and zero-shot
models' prediction files exist, it adds the pre-registered comparisons C1-C4.

The test labels are read once, at the end (Phase 6), and only after every prediction file is
recorded in a commit:

```
python -m ml.evaluation.evaluate --write-hashes   # writes ml/TEST_PREDICTIONS.sha256; commit it
python -m ml.evaluation.evaluate --split test --open-test-labels
```
