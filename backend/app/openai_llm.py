"""Shared LLM client settings (provider, model, timeouts) for all agents.

Uses OpenRouter when ``OPENROUTER_API_KEY`` is set, otherwise OpenAI directly.
"""
from __future__ import annotations

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


def chat_openai_mini(*, temperature: float | None = 0) -> ChatOpenAI:
    settings = get_settings()
    extra: dict = {}
    if settings.openrouter_api_key and not settings.llm_reasoning:
        # Reasoning-capable models (e.g. DeepSeek) think by default: ~4x slower and mostly
        # hidden tokens. Extraction and rewriting here don't need it. Set LLM_REASONING=true
        # for models that make reasoning mandatory.
        extra["extra_body"] = {"reasoning": {"enabled": False}}
    return ChatOpenAI(
        model=settings.llm_model or model_id(DEFAULT_CHAT_MODEL),
        temperature=temperature,
        timeout=OPENAI_REQUEST_TIMEOUT_S,
        max_retries=OPENAI_MAX_RETRIES,
        **client_kwargs(),
        **extra,
    )
