"""Raw resume text -> ResumeInfo. Mirrors the LLM step of app.agents.cv_agent.extract_resume_info,
which only accepts PDFs; the HF dataset provides plain text."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.models.schemas import ResumeInfo

# Copied verbatim from cv_agent.extract_resume_info (inline there, not importable); keep in sync.
_SYSTEM = (
    "Extract all information from the resume text. Be thorough and capture all details. "
    "Include: contact information (email, phone, location), professional links (GitHub, LinkedIn, portfolio/website URLs), "
    "and a short professional summary if one is present. "
    "For education: one entry per degree/program with institution, degree, location, dates "
    "(as written, e.g. 'Expected: September 2027'), grade if stated, and any awards/details. "
    "For experience: include the location of each role if stated. "
    "For leadership roles, extracurricular activities, and volunteer work: one leadership_activities "
    "entry per activity with role, organization, location, dates, a short description, and impact bullets."
)

_CHAIN = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("user", "{text}")]) | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(ResumeInfo)


def extract_resume_info_from_text(text: str) -> ResumeInfo:
    return _CHAIN.invoke({"text": text})
