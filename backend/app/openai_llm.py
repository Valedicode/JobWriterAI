"""Shared LLM client settings (provider, model, timeouts) for all agents.

Uses OpenRouter when ``OPENROUTER_API_KEY`` is set, otherwise OpenAI directly.
Each call site asks for a role (extract, writer, judge, chat) rather than a
model, so the judge can run on a different model than the writer it grades.
"""
from __future__ import annotations

from typing import Literal, get_args

from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.config import get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Default OpenAI httpx client can raise ConnectTimeout / APITimeoutError on slow paths.
OPENAI_REQUEST_TIMEOUT_S = 120.0
OPENAI_MAX_RETRIES = 2


def client_kwargs() -> dict[str, str]:
    """``api_key`` (and ``base_url`` for OpenRouter) shared by ChatOpenAI and OpenAI."""
    settings = get_settings()
    if settings.openrouter_api_key:
        return {"api_key": settings.openrouter_api_key, "base_url": OPENROUTER_BASE_URL}
    if settings.openai_api_key:
        return {"api_key": settings.openai_api_key}
    raise ValueError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in backend/.env")


def model_id(name: str) -> str:
    """OpenRouter namespaces models by vendor (``openai/gpt-4o-mini``)."""
    return f"openai/{name}" if get_settings().openrouter_api_key else name


def openai_client() -> OpenAI:
    return OpenAI(**client_kwargs())


LLMRole = Literal["extract", "writer", "judge", "chat"]
LLM_ROLES: tuple[LLMRole, ...] = get_args(LLMRole)


def model_for(role: LLMRole) -> str:
    """The model a role runs on: its LLM_MODEL_<ROLE> override, else LLM_MODEL,
    else the default chat model."""
    if role not in LLM_ROLES:
        raise ValueError(f"Unknown LLM role {role!r}; expected one of {LLM_ROLES}")
    settings = get_settings()
    return getattr(settings, f"llm_model_{role}") or settings.llm_model or model_id(DEFAULT_CHAT_MODEL)


def provider_slugs() -> list[str]:
    """The OpenRouter providers LLM_PROVIDER pins the chat models to (empty: any host)."""
    pinned = get_settings().llm_provider or ""
    return [slug.strip() for slug in pinned.split(",") if slug.strip()]


def chat_model(role: LLMRole, *, temperature: float | None = 0) -> ChatOpenAI:
    settings = get_settings()
    body: dict = {}
    if settings.openrouter_api_key and not settings.llm_reasoning:
        # Reasoning-capable models (e.g. DeepSeek) think by default: ~4x slower and mostly
        # hidden tokens. Extraction and rewriting here don't need it. Set LLM_REASONING=true
        # for models that make reasoning mandatory.
        body["reasoning"] = {"enabled": False}
    if settings.openrouter_api_key and provider_slugs():
        # Open-weight models have dozens of hosts on OpenRouter, with different quantisations
        # and data terms. A pinned request fails rather than silently moving to another host.
        body["provider"] = {"order": provider_slugs(), "allow_fallbacks": False}
    return ChatOpenAI(
        model=model_for(role),
        temperature=temperature,
        timeout=OPENAI_REQUEST_TIMEOUT_S,
        max_retries=OPENAI_MAX_RETRIES,
        **client_kwargs(),
        **({"extra_body": body} if body else {}),
    )
