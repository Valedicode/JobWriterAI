"""chat_model pins OpenRouter to the providers in LLM_PROVIDER, without fallback to other hosts,
and embeddings call OpenAI itself when its key is set."""

from unittest.mock import patch

from app import openai_llm
from app.config import Settings


def _settings(**settings):
    return patch.object(openai_llm, "get_settings", return_value=Settings(_env_file=None, **settings))


def _chat_model(**settings):
    with _settings(**settings):
        return openai_llm.chat_model("judge")


def test_embeddings_call_openai_itself_when_its_key_is_set():
    with _settings(openrouter_api_key="or-key", openai_api_key="oa-key"):
        client = openai_llm.embeddings_client()
        assert client.api_key == "oa-key" and "openrouter" not in str(client.base_url)
        assert openai_llm.embedding_model_id("text-embedding-3-small") == "text-embedding-3-small"


def test_embeddings_go_through_openrouter_without_an_openai_key():
    with _settings(openrouter_api_key="or-key", openai_api_key=None):
        assert "openrouter.ai" in str(openai_llm.embeddings_client().base_url)
        assert openai_llm.embedding_model_id("text-embedding-3-small") == "openai/text-embedding-3-small"


def test_pinned_providers_are_sent_without_fallback():
    model = _chat_model(openrouter_api_key="k", llm_model="deepseek/deepseek-v4.1-flash-1017",
                        llm_provider="deepseek, together")
    assert model.extra_body == {"reasoning": {"enabled": False},
                                "provider": {"order": ["deepseek", "together"], "allow_fallbacks": False}}


def test_without_a_pin_openrouter_picks_the_host():
    model = _chat_model(openrouter_api_key="k", llm_model="deepseek/deepseek-v4.1-flash-1017")
    assert model.extra_body == {"reasoning": {"enabled": False}}


def test_a_direct_openai_key_gets_no_openrouter_fields():
    model = _chat_model(openrouter_api_key=None, openai_api_key="k", llm_provider="deepseek", llm_reasoning=True)
    assert not model.extra_body
