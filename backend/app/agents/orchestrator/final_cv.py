"""
The CV a session's PDF was generated from: the one a chat edit changes.

Job tailoring renders ``tailored_cv_data`` (step 6); the CV review renders the
revised ``cv_data`` (step 7). Before a CV PDF exists there is nothing to edit
this way: during the review steps changes go through the step's own gate.
"""

from __future__ import annotations

from typing import Any

from app.services.bullet_index import index_cv_bullets


def final_cv(state: dict[str, Any] | None) -> tuple[str, dict[str, Any]] | None:
    """(state key, CV) of the CV the session's PDF was made from, or None."""
    state = state or {}
    files = state.get("generated_files") or []
    if not any(isinstance(f, dict) and f.get("file_type") == "cv" for f in files):
        return None
    key = {"job_tailoring": "tailored_cv_data", "cv_review": "cv_data"}.get(state.get("flow") or "")
    cv = state.get(key) if key else None
    return (key, cv) if isinstance(cv, dict) else None


def final_cv_lines(state: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Every editable line of the final CV as (line id, text), in CV order."""
    found = final_cv(state)
    if found is None:
        return []
    return [(line_id, ref["text"]) for line_id, ref in index_cv_bullets(found[1]).items()]
