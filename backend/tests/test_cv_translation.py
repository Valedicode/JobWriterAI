"""
Translator: the whole tailored CV in the plan's language, reviewed first.

Regression: the plan chose German, only the rewritten summary came out German,
and education, titles and bullets stayed English.

- every free-text field is translated; proper nouns are not;
- a translation that adds or drops a technology, number or year is rejected
  for that field, which keeps its original text;
- the approve_translation gate uses it, keeps the original language, or
  translates again with the user's notes; a failed call can be retried.

Run with::

    cd backend
    pytest tests/test_cv_translation.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from app.agents.translator_agent import (
    TranslatedFields,
    apply_translation,
    check_translation,
    translatable_fields,
    translate_cv,
    translate_cv_fields,
)

from tests.test_orchestrator_job_tailoring import (  # noqa: F401  (fixtures)
    CV,
    JOB,
    PLAN,
    _fake_tool,
    _interrupt_payload,
    graph,
    patched_tools,
)

TCV = {
    "name": "Alex Example",
    "email": "alex@example.com",
    "summary": "Computer science student with Python experience.",
    "skills": ["Python"],
    "experience": [{"position": "Working Student", "company": "Fraunhofer FIT", "location": "Aachen, Germany",
                    "duration": "Jan 2024 -- Present", "responsibilities": ["Built a Django app used by 30 people"]}],
    "education": [{"institution": "Gymnasium Aachen", "degree": "Higher School Certificate",
                   "dates": "2019", "grade": "1.6 (German System)", "details": []}],
    "projects": [{"name": "Query Lab", "description": "A SQL playground", "technologies": ["React"], "outcomes": []}],
    "languages": [{"language": "German", "fluency": "native"}],
}


def test_free_text_fields_are_translated_and_proper_nouns_are_not():
    paths = [f["path"] for f in translatable_fields(TCV)]
    assert paths == [
        "summary",
        "experience.0.position", "experience.0.location", "experience.0.duration",
        "experience.0.responsibilities.0",
        "education.0.degree", "education.0.dates", "education.0.grade",
        "projects.0.description",
        "languages.0.language", "languages.0.fluency",
    ]
    texts = " ".join(f["text"] for f in translatable_fields(TCV))
    for proper in ("Fraunhofer FIT", "Gymnasium Aachen", "Query Lab", "alex@example.com", "Alex Example"):
        assert proper not in texts


def test_apply_translation_writes_each_path():
    out = apply_translation(TCV, {"education.0.degree": "Abitur", "experience.0.responsibilities.0": "x"})
    assert out["education"][0]["degree"] == "Abitur"
    assert out["experience"][0]["responsibilities"] == ["x"]
    assert TCV["education"][0]["degree"] == "Higher School Certificate"  # the input is untouched


@pytest.mark.parametrize("original, translated, problem", [
    ("1.6 (German System)", "Note: 1,6", None),                       # decimal comma is the same number
    ("Built a Django app used by 30 people", "Django-App für 30 Personen entwickelt", None),
    ("Led two teams", "Leitete zwei Teams", None),
    ("Built a Django app", "Eine Flask-App entwickelt", "technologies changed (django, flask)"),
    ("Used by 30 people", "Von 300 Personen genutzt", "a number changed"),
    ("Since 2021", "Seit 2022", "a year changed"),
    ("Something", "", "the translation is empty"),
])
def test_each_field_must_keep_its_facts(original, translated, problem):
    assert check_translation(original, translated) == problem


def test_a_field_that_changes_a_fact_keeps_its_original_text():
    fields = [
        {"path": "education.0.degree", "text": "Abitur"},
        {"path": "experience.0.responsibilities.0", "text": "Eine Flask-App für 30 Personen entwickelt"},
    ]
    cv, rows = translate_cv(TCV, fields)
    assert cv["education"][0]["degree"] == "Abitur"
    assert cv["experience"][0]["responsibilities"][0] == "Built a Django app used by 30 people"
    by_path = {r["path"]: r for r in rows}
    assert by_path["experience.0.responsibilities.0"]["problem"].startswith("technologies changed")
    assert by_path["summary"]["problem"] == "not translated"


def test_the_tool_reports_failures_as_errors():
    class _Boom:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("quota")))

    with patch("app.agents.translator_agent.chat_model", return_value=_Boom()):
        out = json.loads(translate_cv_fields.invoke({"fields_json": "[]", "language": "german"}))
    assert out == {"error": "quota"}


def test_the_tool_returns_the_models_fields():
    class _Model:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: TranslatedFields(fields=[{"path": "summary", "text": "Student"}]))

    with patch("app.agents.translator_agent.chat_model", return_value=_Model()):
        out = json.loads(translate_cv_fields.invoke({"fields_json": "[]", "language": "german"}))
    assert out == {"fields": [{"path": "summary", "text": "Student"}]}


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

# Bullet 0 was rewritten earlier in the flow to mention Python, so its
# translation has to keep Python (the fact check rejects it otherwise).
GERMAN = {
    "experience.0.position": "Ingenieur",
    "experience.0.responsibilities.0": "Entwarf und validierte den ersten veröffentlichten Algorithmus in Python.",
    "experience.0.responsibilities.1": "Schrieb technische Notizen",
}


def _translator(calls: list, error: str | None = None):
    def translate(kwargs):
        calls.append(kwargs)
        if error and len(calls) == 1:
            return json.dumps({"error": error})
        fields = json.loads(kwargs["fields_json"])
        return json.dumps({"fields": [{"path": f["path"], "text": GERMAN.get(f["path"], f["text"])} for f in fields]})
    return _fake_tool(translate)


def _to_translation_gate(graph, thread, calls, error=None):  # noqa: F811
    german_plan = {**PLAN, "cv_language": "german"}
    patches = [
        patch("app.agents.orchestrator.nodes_job_tailoring.plan_tailoring_strategy",
              _fake_tool(lambda _: json.dumps(german_plan))),
        patch("app.agents.orchestrator.nodes_job_tailoring.translate_cv_fields", _translator(calls, error)),
    ]
    for p in patches:
        p.start()
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": JOB}, config=config)
    result = None
    for _ in range(3):  # score + plan, selection, rewrite
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    return config, result, patches


@pytest.fixture
def translating(graph, patched_tools):  # noqa: F811
    started: list = []

    def go(thread, calls, error=None):
        config, result, patches = _to_translation_gate(graph, thread, calls, error)
        started.extend(patches)
        return config, result

    yield go
    for p in started:
        p.stop()


def test_the_translation_is_reviewed_and_then_rendered(graph, patched_tools, translating):  # noqa: F811
    calls: list = []
    config, result = translating("translate-approve", calls)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_translation"
    assert payload["preview"]["target_language"] == "German"
    assert {c["path"] for c in payload["preview"]["changes"]} == set(GERMAN)
    assert calls[0]["language"] == "german"

    rendered: dict = {}

    def render(kwargs):
        rendered["cv"] = json.loads(kwargs["cv_json"])
        return f"CV PDF generated successfully! The file '{kwargs['output_filename']}' is ready for download."

    with patch("app.agents.orchestrator.nodes_job_tailoring.generate_cv_pdf", _fake_tool(render)):
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert _interrupt_payload(result)["step"] == "cover_letter_language"
    assert rendered["cv"]["experience"][0]["responsibilities"][1] == "Schrieb technische Notizen"
    assert graph.get_state(config).values["cv_data"]["experience"][0]["position"] == "Engineer"  # yours untouched


def test_rejecting_keeps_the_original_language(graph, patched_tools, translating):  # noqa: F811
    config, _ = translating("translate-reject", [])
    graph.invoke(Command(resume={"action": "reject"}), config=config)
    tailored = graph.get_state(config).values["tailored_cv_data"]
    assert tailored["experience"][0]["position"] == "Engineer"


def test_edit_translates_again_with_the_users_notes(graph, patched_tools, translating):  # noqa: F811
    calls: list = []
    config, _ = translating("translate-edit", calls)
    result = graph.invoke(Command(resume={"action": "edit", "feedback": "Use Werkstudent"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_translation"
    assert len(calls) == 2 and calls[1]["feedback"] == "- Use Werkstudent"


def test_a_failed_translation_can_be_retried(graph, patched_tools, translating):  # noqa: F811
    calls: list = []
    config, result = translating("translate-retry", calls, error="provider timeout")
    payload = _interrupt_payload(result)
    assert payload["step"] == "step_failed" and payload["preview"]["failed_step"] == "Translation"
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_translation"
    assert len(calls) == 2
