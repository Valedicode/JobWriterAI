"""
Rewrite verifier: rewrites are fact-checked against the ledger before review.

Unit checks (no graph): an invented technology, number or year blocks; a
technology shown elsewhere on the CV but not cited, a cliché, a length blow-up
or a language switch (with no chosen CV language) only warns, while a rewrite in another language than the chosen one blocks; citations and number words are understood;
the summary is checked against the whole CV; user-written text is skipped.

In the graph: a blocked pass is rewritten once with the problems as feedback
and the passing bullets locked; still blocked, the verifier_failed gate shows
the problems, and approving there keeps the original wording - a blocked
rewrite never reaches the tailored CV.

Run with::

    cd backend
    pytest tests/test_rewrite_verifier.py -v
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.types import Command

from app.agents.orchestrator.nodes_job_tailoring import _index_cv_bullets
from app.services.fact_ledger import build_fact_ledger
from app.services.verifier import normalize_metric, retry_feedback, verify_rewrites

from tests.test_orchestrator_job_tailoring import (  # noqa: F401  (fixtures)
    CV,
    JOB,
    _fake_tool,
    _interrupt_payload,
    graph,
    patched_tools,
)

VCV = {
    "summary": "Backend engineer working with Python.",
    "skills": ["Python", "Docker"],
    "experience": [{
        "position": "Engineer", "company": "Analytical Engines Ltd.", "duration": "2021 - 2023",
        "responsibilities": ["Built a REST API in Django for two teams", "Cut hosting costs by 30% in 2022"],
    }],
    "projects": [{"name": "Query Lab", "technologies": ["PostgreSQL"], "outcomes": []}],
}
VINDEX = _index_cv_bullets(VCV)
VLEDGER = build_fact_ledger(VCV, VINDEX)
API, COSTS = "experience.0.responsibilities.0", "experience.0.responsibilities.1"


def _verify(*bullets: dict[str, Any], summary: str = "", user_written=()) -> Any:
    return verify_rewrites(
        {"rewritten_bullets": list(bullets), "updated_summary": summary},
        VLEDGER, VINDEX, cv_summary=VCV["summary"], user_written_ids=user_written,
    )


def _checks(result) -> list[tuple[str, str]]:
    return [(i.check, i.severity) for i in result.issues]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def test_an_invented_technology_blocks():
    result = _verify({"bullet_id": API, "rewritten": "Built a Django REST API on Kubernetes for two teams"})
    assert not result.passed
    assert _checks(result) == [("unknown_entity", "block")]
    assert result.issues[0].offending_span == "kubernetes"
    assert result.blocked_ids() == [API]


def test_a_changed_number_blocks():
    result = _verify({"bullet_id": COSTS, "rewritten": "Cut hosting costs by 45% in 2022"})
    assert _checks(result) == [("new_metric", "block")]


def test_a_changed_date_blocks():
    result = _verify({"bullet_id": COSTS, "rewritten": "Cut hosting costs by 30% in 2024"})
    assert _checks(result) == [("changed_dates", "block")]


def test_a_cliche_only_warns():
    result = _verify({"bullet_id": COSTS, "rewritten": "Results-driven: cut hosting costs by 30% in 2022"})
    assert result.passed
    assert _checks(result) == [("banned_phrase", "warn")]


def test_a_skill_from_elsewhere_on_the_cv_needs_a_citation():
    uncited = _verify({"bullet_id": API, "rewritten": "Built a Dockerised Django REST API for two teams, in Docker"})
    assert uncited.passed and _checks(uncited) == [("missing_citation", "warn")]
    cited = _verify({"bullet_id": API, "rewritten": "Built a Django REST API in Docker for two teams",
                     "cited_fact_ids": ["skill.docker"]})
    assert cited.passed and cited.issues == []


def test_citing_a_fact_that_does_not_exist_warns():
    result = _verify({"bullet_id": API, "rewritten": "Built a Django REST API for two teams",
                      "cited_fact_ids": ["skill.rust"]})
    assert result.passed and _checks(result) == [("missing_citation", "warn")]


def test_rewording_numbers_and_number_words_is_fine():
    result = _verify(
        {"bullet_id": API, "rewritten": "Designed a Django REST API serving 2 teams"},
        {"bullet_id": COSTS, "rewritten": "Reduced hosting costs by 30 percent in 2022"},
    )
    assert result.passed and result.issues == []
    assert result.preservation_score == 1.0


@pytest.mark.parametrize("a, b", [("1,200", "1200"), ("1.200", "1200"), ("30 %", "30%"), ("30 percent", "30%")])
def test_metric_normalisation(a, b):
    assert normalize_metric(a) == normalize_metric(b)


def test_the_summary_is_checked_against_the_whole_cv():
    ok = _verify(summary="Python and Docker engineer who cut hosting costs by 30%, working since 2021.")
    assert ok.passed and ok.issues == []
    bad = _verify(summary="Python engineer with Kubernetes experience.")
    assert not bad.passed and bad.blocked_ids() == ["summary"]


def test_user_written_and_unchanged_bullets_are_not_checked():
    result = _verify(
        {"bullet_id": API, "rewritten": "Built a Django API on Kubernetes for two teams"},
        {"bullet_id": COSTS, "rewritten": VCV["experience"][0]["responsibilities"][1]},
        user_written=[API],
    )
    assert result.passed and result.issues == []


def test_an_unplaceable_rewrite_warns():
    result = _verify({"bullet_id": "experience.9.responsibilities.0", "original": "x", "rewritten": "Wrote docs"})
    assert ("structure_changed", "warn") in _checks(result)


def test_length_and_language_warnings():
    long = "Built a REST API in Django for two teams " + "and kept refining the documentation " * 6
    assert ("length_blowup", "warn") in _checks(_verify({"bullet_id": API, "rewritten": long}))
    german = "Entwicklung einer REST API mit Django für die zwei Teams und den Betrieb"
    assert ("language_mismatch", "warn") in _checks(_verify({"bullet_id": API, "rewritten": german}))


GERMAN_BULLET = "Entwicklung einer REST API mit Django für die zwei Teams und den Betrieb"


def test_a_rewrite_in_another_language_than_the_chosen_one_blocks():
    """A German posting must not turn an English CV German: with a chosen CV language a rewrite in
    another one blocks (so it is written again), where before it only warned."""
    def verify(language, **bullet):
        return verify_rewrites(
            {"rewritten_bullets": [{"bullet_id": API, **bullet}], "updated_summary": ""},
            VLEDGER, VINDEX, cv_summary=VCV["summary"], cv_language=language,
        )

    blocked = verify("english", rewritten=GERMAN_BULLET)
    assert ("language_mismatch", "block") in _checks(blocked) and not blocked.passed
    assert "German" in blocked.issues[0].detail and "English" in blocked.issues[0].detail
    assert "in the CV's language" in retry_feedback(blocked)
    assert verify("german", rewritten=GERMAN_BULLET).passed
    # Text too short to tell is not a switch; with no choice the old warning stays.
    assert verify("english", rewritten="Built a REST API in Django for both teams").passed
    assert ("language_mismatch", "warn") in _checks(verify(None, rewritten=GERMAN_BULLET))


def test_a_summary_in_the_wrong_language_blocks_too():
    german = "Backend-Entwickler mit Python und Docker, der die Teams bei der Arbeit unterstützt und für den Betrieb sorgt"
    result = verify_rewrites(
        {"rewritten_bullets": [], "updated_summary": german}, VLEDGER, VINDEX,
        cv_summary=VCV["summary"], cv_language="english",
    )
    assert ("language_mismatch", "block") in _checks(result)


def test_the_rewrite_prompt_names_the_chosen_language():
    from app.agents.writer_agent import _language_rule

    assert _language_rule("english").startswith("English")
    assert _language_rule("german").startswith("German")
    assert _language_rule("") == "the language the CV is written in"


def test_retry_feedback_names_each_blocked_claim():
    result = _verify({"bullet_id": API, "rewritten": "Built a Django REST API on Kubernetes for two teams"})
    text = retry_feedback(result)
    assert API in text and "kubernetes" in text


# --------------------------------------------------------------------------- #
# In the graph
# --------------------------------------------------------------------------- #

BULLET = "experience.0.responsibilities.0"  # "Designed the first algorithm" in the shared CV
SELECTED = {
    "selected_bullets": [{"section": "experience", "original_text": "Designed the first algorithm",
                          "relevance_score": 0.9, "reason": "", "bullet_id": BULLET}],
    "section_order": [], "sections_to_emphasize": [], "items_to_de_emphasize": [],
}
CLEAN = "Designed and validated the first algorithm"
FABRICATED = "Designed the first algorithm and deployed it on Kubernetes"


class _Rewrites:
    """Returns the given rewrite texts pass by pass and records each call."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, kwargs: dict[str, Any]) -> str:
        self.calls.append(kwargs)
        locked = {b["bullet_id"]: b for b in json.loads(kwargs["locked_bullets_json"])}
        text = self.texts[min(len(self.calls), len(self.texts)) - 1]
        bullet = locked.get(BULLET) or {"original": "Designed the first algorithm", "rewritten": text,
                                        "confidence": 0.9, "keywords_added": [], "bullet_id": BULLET}
        return json.dumps({"rewritten_bullets": [bullet], "updated_summary": "", "keywords_inserted": []})


@pytest.fixture
def run(graph, patched_tools):  # noqa: F811
    """Start a run and approve up to the rewrite step with the given rewrites."""
    def start(*texts: str):
        rewrites = _Rewrites(*texts)
        patches = [
            patch("app.agents.orchestrator.nodes_job_tailoring.select_prioritize_content",
                  _fake_tool(lambda _: json.dumps(SELECTED))),
            patch("app.agents.orchestrator.nodes_job_tailoring.rewrite_enhance_content", rewrites),
        ]
        for p in patches:
            p.start()
        config = {"configurable": {"thread_id": f"verify-{len(texts)}-{texts[0][:12]}-{id(rewrites)}"}}
        graph.invoke({"flow": "job_tailoring", "cv_language": "english", "cv_data": CV, "job_data": JOB}, config=config)
        graph.invoke(Command(resume={"action": "approve"}), config=config)  # score
        result = graph.invoke(Command(resume={"action": "approve"}), config=config)  # selection -> rewrite
        start.cleanup.extend(patches)
        return config, result, rewrites

    start.cleanup = []
    yield start
    for p in start.cleanup:
        p.stop()


def test_clean_rewrites_go_straight_to_review(graph, run):  # noqa: F811
    config, result, rewrites = run(CLEAN)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    assert len(rewrites.calls) == 1
    assert payload["preview"]["rewritten_bullets"][0]["verifier_notes"] == []
    assert graph.get_state(config).values["verifier_result"]["passed"] is True


def test_the_rewrite_prompt_gets_the_citable_facts(run):
    _, _, rewrites = run(CLEAN)
    facts = rewrites.calls[0]["facts_block"]
    assert "[skill.python] Python" in facts
    assert BULLET not in facts  # bullets are already in the prompt with their ids


def test_a_blocked_pass_is_rewritten_once_with_the_problems(graph, run):  # noqa: F811
    config, result, rewrites = run(FABRICATED, CLEAN)
    assert _interrupt_payload(result)["step"] == "approve_rewrite"
    assert len(rewrites.calls) == 2
    feedback = json.loads(rewrites.calls[1]["tailoring_strategy_json"])["user_feedback"][-1]
    assert feedback["stage"] == "verifier" and "kubernetes" in feedback["text"]
    values = graph.get_state(config).values
    assert values["rewritten_content"]["rewritten_bullets"][0]["rewritten"] == CLEAN
    assert values["verify_attempt"] is None
    assert any("Rewriting those again" in m.content for m in values["messages"])


def test_still_blocked_after_the_retry_stops_at_the_verifier_gate(run):
    _, result, rewrites = run(FABRICATED, FABRICATED)
    payload = _interrupt_payload(result)
    assert payload["step"] == "verifier_failed"
    assert len(rewrites.calls) == 2
    assert any("kubernetes" in p for p in payload["preview"]["problems"])
    assert payload["preview"]["blocked_rewrites"][0]["rewritten"] == FABRICATED


def test_approving_at_the_verifier_gate_keeps_the_original_wording(graph, run):  # noqa: F811
    config, _, _ = run(FABRICATED, FABRICATED)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    bullet = payload["preview"]["rewritten_bullets"][0]
    assert bullet["rewritten"] == "Designed the first algorithm"
    graph.invoke(Command(resume={"action": "approve"}), config=config)
    tailored = graph.get_state(config).values["tailored_cv_data"]
    assert tailored["experience"][0]["responsibilities"][0] == "Designed the first algorithm"
    assert "Kubernetes" not in json.dumps(tailored)


def test_editing_at_the_verifier_gate_rewrites_with_a_fresh_check(graph, run):  # noqa: F811
    config, _, rewrites = run(FABRICATED, FABRICATED, CLEAN)
    result = graph.invoke(Command(resume={"action": "edit", "feedback": "No new tools"}), config=config)
    assert _interrupt_payload(result)["step"] == "approve_rewrite"
    assert len(rewrites.calls) == 3
    feedback = json.loads(rewrites.calls[2]["tailoring_strategy_json"])["user_feedback"][-1]
    assert feedback == {"stage": "approve_rewrite", "text": "No new tools"}


def test_rejecting_at_the_verifier_gate_stops(graph, run):  # noqa: F811
    config, _, _ = run(FABRICATED, FABRICATED)
    result = graph.invoke(Command(resume={"action": "reject"}), config=config)
    assert "__interrupt__" not in result
    assert graph.get_state(config).values.get("tailored_cv_data") is None


def test_the_users_own_wording_is_not_blocked(graph, run):  # noqa: F811
    config, _, rewrites = run(CLEAN)
    custom = "Designed the first algorithm, later ported to Kubernetes"
    result = graph.invoke(Command(resume={
        "action": "edit", "feedback": "keep mine",
        "decisions": {BULLET: {"choice": "custom", "text": custom}},
    }), config=config)
    payload = _interrupt_payload(result)
    assert payload["step"] == "approve_rewrite"
    assert payload["preview"]["rewritten_bullets"][0]["rewritten"] == custom
    assert len(rewrites.calls) == 2
