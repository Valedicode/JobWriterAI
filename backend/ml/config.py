from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = BACKEND_DIR / "data" / "ml"  # git-ignored via backend/data/

HF_DATASET = "cnamuangtoun/resume-job-description-fit"
HF_CACHE = ARTIFACTS / "hf_eval_cache"
TEACHER_LABELS = ARTIFACTS / "teacher_labels"
CHECKPOINTS = ARTIFACTS / "checkpoints"

SEED = 42
