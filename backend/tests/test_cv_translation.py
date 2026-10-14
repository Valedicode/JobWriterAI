"""
CV language: asked after scoring, and the whole CV translated (and reviewed)
into it before the plan and rewriting.

Regression: the plan chose German, a source CV that was already mixed (German
summary, English entries) looked German enough to skip translation, and the
result was half German, half English.

- every free-text field is translated; proper nouns are not;
- a translation that adds or drops a technology, number or year is rejected
  for that field, which keeps its original text;
- the cv_language gate asks in the posting's language first; picking a
  language the CV isn't (wholly) in translates it, reviewed at
  approve_translation: use it, keep the CV as it is, or translate again;
  a failed call can be retried. Every later step then sees one language.

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
    needs_translation,
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
# Which fields need it
# --------------------------------------------------------------------------- #

def test_a_mixed_cv_needs_translation_either_way():
    mixed = {**TCV, "summary": "Informatik-Student mit Erfahrung in der Webentwicklung und in der Teamarbeit."}
    assert needs_translation(mixed, "german") and needs_translation(mixed, "english")
    assert not needs_translation(TCV, "english")


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

GERMAN_POSTING = {**JOB, "posting_language": "de"}
GERMAN = {
    "experience.0.position": "Ingenieur",
    "experience.0.responsibilities.0": "Entwarf den ersten Algorithmus",
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


@pytest.fixture
def run(graph, patched_tools):  # noqa: F811
    """Start a run (no CV language preset) with the translator stubbed."""
    started: list = []

    def go(thread, job=GERMAN_POSTING, calls=None, error=None):
        p = patch("app.agents.orchestrator.nodes_job_tailoring.translate_cv_fields",
                  _translator(calls if calls is not None else [], error))
        p.start()
        started.append(p)
        config = {"configurable": {"thread_id": thread}}
        return config, graph.invoke({"flow": "job_tailoring", "cv_data": CV, "job_data": job}, config=config)

    yield go
    for p in started:
        p.stop()


def _choose(graph, config, language):  # noqa: F811
    return graph.invoke(Command(resume={"action": "choose", "choice": language}), config=config)


def test_the_cv_language_is_asked_with_the_postings_language_first(graph, run):  # noqa: F811
    _, result = run("lang-ask")
    payload = _interrupt_payload(result)
    assert payload["step"] == "cv_language" and payload["kind"] == "choice"
    assert payload["choices"] == ["german", "english"]
    assert "The posting is in German" in payload["narration"]
    _, result = run("lang-ask-en", job={**JOB, "posting_language": "en"})
    assert _interrupt_payload(result)["choices"] == ["english", "german"]


def test_choosing_german_translates_the_cv_before_the_plan(graph, run):  # noqa: F811
    calls: list = []
    config, _ = run("lang-de", calls=calls)
    payload = _interrupt_payload(_choose(graph, config, "german"))
    assert payload["step"] == "approve_translation"
    assert {c["path"] for c in payload["preview"]["changes"]} == set(GERMAN)
    assert calls[0]["language"] == "german"

    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert _interrupt_payload(result)["step"] == "present_score"  # then the plan and the score review
    values = graph.get_state(config).values
    assert values["cv_language"] == "german"
    assert values["cv_data"]["experience"][0]["position"] == "Ingenieur"
    assert values["bullet_index"]["experience.0.responsibilities.1"]["text"] == "Schrieb technische Notizen"
    assert values["fact_ledger"]["experience.0.responsibilities.1"]["value"] == "Schrieb technische Notizen"
    assert values["original_cv_data"]["experience"][0]["position"] == "Engineer"  # as extracted
    assert values["tailoring_strategy"]["plan"]["cv_language"] == "german"  # the user's choice wins


def test_a_cv_already_in_the_chosen_language_is_not_translated(graph, run):  # noqa: F811
    calls: list = []
    config, _ = run("lang-en", calls=calls)
    assert _interrupt_payload(_choose(graph, config, "english"))["step"] == "present_score"
    assert calls == []
    assert graph.get_state(config).values["cv_language"] == "english"


def test_rejecting_the_translation_keeps_the_cv(graph, run):  # noqa: F811
    config, _ = run("lang-reject")
    _choose(graph, config, "german")
    result = graph.invoke(Command(resume={"action": "reject"}), config=config)
    assert _interrupt_payload(result)["step"] == "present_score"
    values = graph.get_state(config).values
    assert values["cv_data"]["experience"][0]["position"] == "Engineer"
    assert values["cv_language"] == "english"


def test_edit_translates_again_with_the_users_notes(graph, run):  # noqa: F811
    calls: list = []
    config, _ = run("lang-edit", calls=calls)
    _choose(graph, config, "german")
    result = graph.invoke(Command(resume={"action": "edit", "feedback": "Use Werkstudent"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_translation"
    assert len(calls) == 2 and calls[1]["feedback"] == "- Use Werkstudent"


def test_a_failed_translation_can_be_retried(graph, run):  # noqa: F811
    calls: list = []
    config, _ = run("lang-retry", calls=calls, error="provider timeout")
    payload = _interrupt_payload(_choose(graph, config, "german"))
    assert payload["step"] == "step_failed" and payload["preview"]["failed_step"] == "Translation"
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_translation"
    assert len(calls) == 2


def test_the_language_is_asked_once_per_session(graph, run):  # noqa: F811
    from app.agents.orchestrator.nodes_job_tailoring import route_after_score

    config, _ = run("lang-once")
    _choose(graph, config, "english")
    assert route_after_score(graph.get_state(config).values) == "jt_step3_decide_strategy"
