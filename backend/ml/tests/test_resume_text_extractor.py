"""The teacher's resume extraction must use the app's current CV prompt, not a copy.

Run from backend/:  python -m pytest ml/tests
"""
from ml.labeling.resume_text_extractor import app_resume_prompt


def test_reads_the_whole_current_app_prompt():
    prompt = app_resume_prompt()
    assert prompt.startswith("Extract all information from the resume text.")
    # Rules the app added after the first copy was made; a stale copy would miss them.
    assert "spoken languages" in prompt
    assert "without the CV's category" in prompt
    assert "Contact fields hold text only" in prompt


def test_the_teacher_extraction_runs_the_apps_checks_and_reports_usage():
    """Since #71 the app removes extraction slips the text contradicts; the teacher must too."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from langchain_core.runnables import RunnableLambda

    from app.models.schemas import ResumeInfo
    from ml.labeling import resume_text_extractor

    bullet = "Developed a management website with Django and PostgreSQL for the institute staff."
    text = ("Activities\nInvestment Club\nProject Manager\nLed a student team.\n"
            "Experience\nFraunhofer IPT\nIntern\n• " + bullet + "\n")
    parsed = ResumeInfo(
        name="Ada", email="a@b.c", phone="1", skills=[], education=[], projects=[],
        leadership_activities=[{"organization": "Investment Club", "role": "Project Manager",
                                "description": "Led a student team.", "highlights": [bullet]}],
        experience=[{"company": "Fraunhofer IPT", "position": "Intern", "responsibilities": [bullet]}],
    )
    raw = SimpleNamespace(usage_metadata={"input_tokens": 1200, "output_tokens": 300})

    class _Stub:
        def with_structured_output(self, schema, include_raw=False):
            return RunnableLambda(lambda _: {"raw": raw, "parsed": parsed, "parsing_error": None})

    with patch.object(resume_text_extractor, "_model", lambda model: _Stub()):
        out = resume_text_extractor.extract_resume(text, model="some/model")
    assert out.cv["leadership_activities"][0]["highlights"] == []
    assert out.cv["experience"][0]["responsibilities"] == [bullet]
    assert len(out.notes) == 1
    assert (out.input_tokens, out.output_tokens) == (1200, 300)
