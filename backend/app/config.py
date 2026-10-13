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
    # OPENAI_API_KEY is still used directly for speech-to-text (OpenRouter has no audio API).
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_model: Optional[str] = None  # chat model override, e.g. "anthropic/claude-sonnet-4.5"
    llm_reasoning: bool = False  # OpenRouter only: allow hidden reasoning tokens (slower, costlier)
    tavily_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Database
    database_url: str = "postgresql://localhost:5432/jobwriterai"

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