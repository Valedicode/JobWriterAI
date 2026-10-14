"""
Model routing: each LLM call site asks for a role, and each role resolves to
its LLM_MODEL_<ROLE> override, else LLM_MODEL, else the default model.

No provider is called; building a ChatOpenAI client doesn't touch the network.

Run with::

    cd backend
    pytest tests/test_model_routing.py -v
"""

from __future__ import annotations

import inspect
import re

import pytest

from app.agents import cv_agent, job_agent, scoring_agent, writer_agent
from app.agents.orchestrator import chat_agent, nodes_cv_review
from app.config import get_settings
from app.openai_llm import LLM_ROLES, chat_model, model_for


@pytest.fixture
def models(monkeypatch):
    """Set every model variable explicitly so backend/.env can't leak in."""
    def configure(base: str = "", **roles: str):
        monkeypatch.setenv("LLM_MODEL", base)
        for role in LLM_ROLES:
            monkeypatch.setenv(f"LLM_MODEL_{role.upper()}", roles.get(role, ""))
        get_settings.cache_clear()
    yield configure
    get_settings.cache_clear()


def test_each_role_resolves_to_its_own_override(models):
    models(base="base/model", extract="x/extract", writer="x/writer", judge="y/judge", chat="x/chat")
    assert {role: model_for(role) for role in LLM_ROLES} == {
        "extract": "x/extract", "writer": "x/writer", "judge": "y/judge", "chat": "x/chat",
    }


def test_roles_without_an_override_use_llm_model(models):
    models(base="base/model", judge="y/judge")
    assert model_for("writer") == "base/model"
    assert model_for("extract") == "base/model"
    assert model_for("judge") == "y/judge"


def test_without_any_model_setting_the_default_is_used(models):
    models()
    assert model_for("chat").endswith("gpt-4o-mini")


def test_unknown_role_is_refused(models):
    models(base="base/model")
    with pytest.raises(ValueError):
        model_for("planner")


def test_chat_model_builds_a_client_on_the_role_model(models):
    models(base="base/model", judge="y/judge")
    assert chat_model("judge", temperature=0).model_name == "y/judge"
    assert chat_model("writer", temperature=0.3).temperature == 0.3


def _roles_requested(module) -> set[str]:
    return set(re.findall(r'chat_model\("(\w+)"', inspect.getsource(module)))


@pytest.mark.parametrize("module, roles", [
    (cv_agent, {"extract"}),
    (job_agent, {"extract"}),
    (scoring_agent, {"judge"}),
    (writer_agent, {"extract", "writer"}),
    (chat_agent, {"chat"}),
    (nodes_cv_review, {"writer"}),
])
def test_call_sites_ask_for_their_role(module, roles):
    assert _roles_requested(module) == roles


def test_writer_prose_and_extraction_use_different_roles():
    source = inspect.getsource(writer_agent)
    for function, role in [
        ("generate_job_summary", "extract"),
        ("analyze_cv_job_alignment", "extract"),
        ("generate_cover_letter_content", "writer"),
        ("select_prioritize_content", "writer"),
        ("rewrite_enhance_content", "writer"),
    ]:
        body = inspect.getsource(getattr(writer_agent, function).func)
        assert f'chat_model("{role}"' in body, function
    assert "chat_openai_mini" not in source
