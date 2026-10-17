"""
Regenerate tests/golden/cv_tex.json: the LaTeX the CV adapter produces for every
fixture in tests/cv_view_fixtures.py under every render option.

Only regenerate when a change to the printed CV is intended, and review the diff.

    cd backend
    python -m tests.golden.regenerate_cv_tex
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.latex_renderer import render_cv_tex
from tests.cv_view_fixtures import CASES, OPTIONS

GOLDEN = Path(__file__).with_name("cv_tex.json")


def render_all() -> dict[str, str]:
    return {f"{case}/{option}": render_cv_tex(cv, **kwargs)
            for case, cv in CASES.items() for option, kwargs in OPTIONS}


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(render_all(), indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"wrote {GOLDEN}")
