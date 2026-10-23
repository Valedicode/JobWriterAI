"""
Application configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # API Keys
    # LLM + embeddings go through OpenRouter when its key is set, otherwise OpenAI.
    # OPENAI_API_KEY is still used directly for speech-to-text (OpenRouter has no audio API) and,
    # when set, for embeddings (OpenRouter refuses OpenAI's models from some regions).
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = None  # chat model override, e.g. "anthropic/claude-sonnet-4.5"
    # Per-role overrides; each falls back to llm_model. Pick the judge from a
    # different model family than the writer so it doesn't grade its own style.
    llm_model_extract: Optional[str] = None  # CV/job parsing, job summary
    llm_model_writer: Optional[str] = None  # selection, rewrites, cover letters
    llm_model_judge: Optional[str] = None  # skill-transferability verdicts
    llm_model_chat: Optional[str] = None  # conversational replies
    llm_reasoning: bool = False  # OpenRouter only: allow hidden reasoning tokens (slower, costlier)
    # OpenRouter only: comma-separated provider slugs (e.g. "deepseek") that may serve the chat
    # models, with no fallback to other hosts. Unset lets OpenRouter pick any host of the model.
    llm_provider: Optional[str] = None
    tavily_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Database: orchestrator sessions (LangGraph checkpoints) are stored here.
    # Unset -> in-memory only, and every restart loses all sessions.
    database_url: Optional[str] = None
    # Sessions (and the CV data in them) are deleted this long after their
    # last activity.
    session_ttl_hours: float = 24.0

    # Application
    app_name: str = "JobWriterAI"
    debug: bool = False
    user_agent: Optional[str] = None

    # CORS
    frontend_url: str = "http://localhost:3000"

    # LangSmith Tracing (optional)
    langchain_tracing_v2: Optional[bool] = None
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields in .env that aren't defined here
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()