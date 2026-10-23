import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Git-ignored via backend/data/. ML_ARTIFACTS points a checkout of the teacher tag (a git worktree)
# at the main checkout's artifacts, so labels and caches from every run land in one place.
ARTIFACTS = Path(os.environ.get("ML_ARTIFACTS") or BACKEND_DIR / "data" / "ml")

HF_DATASET = "cnamuangtoun/resume-job-description-fit"
# The dataset repo's commit when the labels were planned (checked 2026-10-23). Loading this
# revision keeps the rows fixed even if the uploader changes the files.
HF_REVISION = "08978e21714984bb417547d2c0f9b477f5298163"
HF_CACHE = ARTIFACTS / "hf_eval_cache"
TEACHER_LABELS = ARTIFACTS / "teacher_labels"
TEACHER_CACHE = ARTIFACTS / "teacher_cache"
CHECKPOINTS = ARTIFACTS / "checkpoints"

SEED = 42
