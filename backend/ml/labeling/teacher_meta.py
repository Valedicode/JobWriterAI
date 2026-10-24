"""teacher_meta.json: which teacher made a label file (NLP_PROJECT_PLAN.md §5.1).

It is written next to each label file when its run starts. A run that continues a label file checks
that the teacher is still the same, so labels from two teacher versions never end up in one file.

A frozen teacher is a clean checkout of a ``teacher-*`` tag, pinned to OpenRouter providers, with
versioned model ids. Anything else is a trial and is marked ``"frozen": false``.
"""
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import requests

from app.config import get_settings
from app.openai_llm import (DEFAULT_EMBEDDING_MODEL, embedding_model_id, embeddings_direct, model_for,
                            provider_slugs)
from ml.config import BACKEND_DIR, HF_DATASET, HF_REVISION

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
PACKAGES = ("langchain-openai", "langchain-core", "langchain-community", "openai", "pydantic", "numpy", "datasets")
# The fields that define the teacher. A continued run must match all of them.
IDENTITY = ("tag", "commit", "dirty", "route", "providers", "models", "embedding_route", "embedding_model",
            "reasoning", "dataset")
# A versioned id ends in a number (deepseek-v4.1-flash-1017). An alias can be moved to a new model mid-run.
VERSIONED = re.compile(r"-\d+$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=BACKEND_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def teacher_version() -> dict:
    """The checkout's tag (if HEAD is exactly on one), commit, and whether teacher code is modified."""
    return {"tag": _git("describe", "--tags", "--exact-match", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", "app", "ml"))}


def endpoint_prices(model: str, providers: list[str]) -> dict:
    """USD per prompt and per completion token for ``model`` on OpenRouter: the pinned providers'
    endpoints, or every endpoint when nothing is pinned. Several endpoints give the highest price."""
    endpoints = requests.get(ENDPOINTS_URL.format(model=model), timeout=60).json()["data"]["endpoints"]
    chosen = [e for e in endpoints if not providers or e["tag"] in providers]
    if not chosen:
        raise ValueError(f"{model}: no OpenRouter endpoint for {providers}; "
                         f"available: {sorted(e['tag'] for e in endpoints)}")
    return {"prompt": max(float(e["pricing"]["prompt"]) for e in chosen),
            "completion": max(float(e["pricing"]["completion"]) for e in chosen),
            "endpoints": sorted(e["tag"] for e in chosen)}


def _package_versions() -> dict:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def build(split: str, labels: str) -> dict:
    """The metadata of a run about to label ``split`` into the file ``labels``."""
    settings = get_settings()
    route = "openrouter" if settings.openrouter_api_key else "openai"
    models = {"extract": model_for("extract"), "judge": model_for("judge")}
    prices = ({m: endpoint_prices(m, provider_slugs()) for m in sorted(set(models.values()))}
              if route == "openrouter" else {})
    return {**teacher_version(), "route": route, "providers": provider_slugs(), "models": models,
            "embedding_route": "openai" if embeddings_direct() else route,
            "embedding_model": embedding_model_id(DEFAULT_EMBEDDING_MODEL), "temperature": 0,
            "reasoning": settings.llm_reasoning,
            "dataset": {"id": HF_DATASET, "revision": HF_REVISION, "split": split},
            "prices_usd_per_token": prices, "packages": _package_versions(),
            "python": platform.python_version(), "labels": labels, "started": now()}


def unfrozen(meta: dict) -> list[str]:
    """Why ``meta`` does not describe a frozen teacher (empty if it does)."""
    problems = []
    if not (meta["tag"] or "").startswith("teacher-"):
        problems.append("HEAD is not on a teacher-* tag")
    if meta["dirty"]:
        problems.append("backend/app or backend/ml has uncommitted changes")
    if meta["route"] == "openrouter" and not meta["providers"]:
        problems.append("LLM_PROVIDER is not set, so OpenRouter may switch hosts")
    problems += [f"{m} is not a versioned model id" for m in sorted(set(meta["models"].values()))
                 if not VERSIONED.search(m)]
    return problems


def differences(old: dict, new: dict) -> list[str]:
    """The teacher-defining fields in which two runs differ."""
    return [k for k in IDENTITY if old.get(k) != new.get(k)]


def write(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
