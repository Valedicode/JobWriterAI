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

The test split's labels are never read. Train labels are read only for the summary.

**Resuming.** Rerunning the same command continues the file: finished rows are skipped and failed
rows are tried again. A run continuing a file must use the same teacher; anything else is refused.

**Stopping.** A row whose score was reached despite a failed API call is retried (the scorer would
otherwise return a wrong score without saying so). A fatal error (bad key, no credit, a host or
region refused, no endpoint) or 25 failures in a row stops the run with exit code 2; finished rows
are kept.

**Caches.** Extractions and embeddings are kept in `teacher_cache/` across runs. Each distinct text
is extracted once per teacher version; `--no-cache` uses fresh ones for the run.
