"""Synthetic CVs for the page-layout tests (no real person's data)."""

from __future__ import annotations

from typing import Any


def _bullets(prefix: str, n: int, words: int = 22) -> list[str]:
    filler = " ".join(["reliable"] * words)
    return [f"{prefix} {i}: built and shipped a feature, {filler}." for i in range(1, n + 1)]


def cv_with(jobs: int, bullets_per_job: int, projects: int = 2, languages: bool = True) -> dict[str, Any]:
    cv: dict[str, Any] = {
        "name": "Alex Example",
        "email": "alex@example.com",
        "phone": "+49 30 000000",
        "location": "Berlin, Germany",
        "summary": "Backend engineer working with Python and PostgreSQL.",
        "skills": ["Python", "PostgreSQL", "Docker", "FastAPI", "Git"],
        "education": [{"institution": "Example University", "degree": "B.Sc. Computer Science",
                       "location": "Berlin", "dates": "Oct 2018 -- Sep 2021", "grade": "1.7", "details": []}],
        "experience": [
            {"position": f"Engineer {j}", "company": f"Company {j}", "location": "Berlin",
             "duration": f"Jan {2010 + j} -- Dec {2011 + j}",
             "responsibilities": _bullets(f"Job {j} bullet", bullets_per_job)}
            for j in range(1, jobs + 1)
        ],
        "projects": [
            {"name": f"Project {p}", "description": f"Project {p} description.",
             "technologies": ["Python"], "outcomes": [f"Outcome of project {p}"]}
            for p in range(1, projects + 1)
        ],
        "leadership_activities": [],
    }
    if languages:
        cv["languages"] = [{"language": "German", "fluency": "native"}, {"language": "English", "fluency": "C1"}]
    return cv
