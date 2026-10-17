"""Raw resume text -> ResumeInfo, exactly as the app extracts a PDF's text.

app.agents.cv_agent.extract_resume_info only accepts PDFs, and its system prompt is an inline
literal inside that function, so it can't be imported. Copying it drifted once already (the app
later added rules for spoken languages, skill names and contact fields). Instead the prompt is
read from cv_agent.py's source here, so the teacher's resume extraction always matches the app
version it is run against, without changing the app (prior work, see NLP_PROJECT_PLAN.md).

Like the app (since #71), the result is checked against the text it came from
(app.services.cv_extraction_checks): bullets copied into a second entry, repeated grades and job
titles repeated as the company are removed, and each removal is reported.
"""
import ast
import inspect
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agents import cv_agent
from app.config import get_settings
from app.models.schemas import ResumeInfo
from app.openai_llm import OPENAI_MAX_RETRIES, OPENAI_REQUEST_TIMEOUT_S, chat_model, client_kwargs
from app.services.cv_extraction_checks import clean_extracted_cv

PROMPT_PREFIX = "Extract all information from the resume text."
# "direct:openai/<model>" calls OpenAI's own API (OPENAI_API_KEY) instead of
# OpenRouter, for comparing a vendor key against the OpenRouter route.
DIRECT_PREFIX = "direct:"


@dataclass
class Extraction:
    """A checked extraction: the CV, what the checks removed, and the call's token usage."""
    cv: dict
    notes: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@lru_cache(maxsize=1)
def app_resume_prompt() -> str:
    """The system prompt string in cv_agent.extract_resume_info (adjacent literals already joined)."""
    tree = ast.parse(inspect.getsource(cv_agent))
    found = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.startswith(PROMPT_PREFIX)]
    if len(found) != 1:
        raise RuntimeError(f"expected one CV extraction prompt in cv_agent.py, found {len(found)}")
    return found[0]


def catalog_id(model: str) -> str:
    """The OpenRouter id of a model, also for a direct call (for its price)."""
    return model.removeprefix(DIRECT_PREFIX)


def _model(model: str | None) -> ChatOpenAI:
    """The app's extraction model (``chat_model("extract")``, as routed by the settings), or
    ``model`` with exactly the same client settings, for comparing candidate teacher models."""
    if model is None:
        return chat_model("extract", temperature=0)
    if model.startswith(DIRECT_PREFIX):
        vendor, _, name = catalog_id(model).partition("/")
        api_key = get_settings().openai_api_key
        if vendor != "openai" or not name or not api_key:
            raise ValueError(f"{model}: a direct call needs an openai/<model> id and OPENAI_API_KEY in backend/.env")
        # OpenAI's reasoning models take only their default temperature; low
        # effort keeps them close to the other candidates, which run without reasoning.
        return ChatOpenAI(model=name, reasoning_effort="low", timeout=OPENAI_REQUEST_TIMEOUT_S,
                          max_retries=OPENAI_MAX_RETRIES, api_key=api_key)
    extra = {} if get_settings().llm_reasoning else {"extra_body": {"reasoning": {"enabled": False}}}
    return ChatOpenAI(model=model, temperature=0, timeout=OPENAI_REQUEST_TIMEOUT_S,
                      max_retries=OPENAI_MAX_RETRIES, **client_kwargs(), **extra)


def extract_resume(text: str, model: str | None = None) -> Extraction:
    """Extract and check one resume text (raises on a failed or unparseable call)."""
    prompt = ChatPromptTemplate.from_messages([("system", app_resume_prompt()), ("user", "{text}")])
    chain = prompt | _model(model).with_structured_output(ResumeInfo, include_raw=True)
    out = chain.invoke({"text": text})
    if out.get("parsing_error") or out.get("parsed") is None:
        raise ValueError(f"unparseable extraction: {out.get('parsing_error')}")
    usage = getattr(out["raw"], "usage_metadata", None) or {}
    cleaned, notes = clean_extracted_cv(out["parsed"].model_dump(), text)
    return Extraction(ResumeInfo.model_validate(cleaned).model_dump(), notes,
                      usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def extract_resume_info_from_text(text: str) -> ResumeInfo:
    """The checked extraction as a ResumeInfo (the teacher's resume step)."""
    return ResumeInfo.model_validate(extract_resume(text).cv)
