"""
Golden tests for the job-extraction system prompt in ``app.agents.job_agent``.

The extraction step (retrieve_web / retrieve_text) is the only place in the
pipeline that ever sees the posting's raw phrasing - by the time it becomes a
flat JobRequirements object, nuance like "eager to learn X" vs. "must have X"
is gone for good. These tests don't call the LLM; they pin that the system
prompt actually carries the intent-classification and OR-alternative
guidance, so a future edit can't silently drop it and re-introduce the bug
where every skill name mentioned in a posting - optional or not - gets
extracted as a hard requirement.

Run with::

    cd backend
    pytest tests/test_job_agent_extraction_prompt.py -v
"""

from __future__ import annotations

from app.agents.job_agent import _JOB_EXTRACTION_SYSTEM_PROMPT


def test_prompt_teaches_or_alternative_grouping():
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "required_skill_groups" in prompt
    assert "WildFly" in prompt and "Quarkus" in prompt and "Spring Boot" in prompt


def test_prompt_teaches_employer_will_train_softens_requirement():
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "training provided" in prompt
    assert "preferred_skills" in prompt


def test_prompt_teaches_eagerness_and_willingness_softens_requirement():
    """The specific gap this test locks in: 'eager to learn Java' must not be
    extracted as a hard requirement for Java - it's a willingness signal, not
    a possession claim."""
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "eager to learn" in prompt.lower()
    assert "willingness" in prompt.lower()
    assert "curiosity" in prompt.lower()


def test_prompt_teaches_classification_is_by_intent_not_keyword_spotting():
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "INTENT" in prompt
    assert "possess" in prompt.lower()


def test_prompt_default_is_preferred_when_unsure():
    """Overstating a requirement damages the score more than understating one
    - the prompt must bias toward preferred_skills on ambiguity."""
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "genuinely unsure" in prompt.lower()
    assert "prefer `preferred_skills`" in prompt or "prefer preferred_skills" in prompt


def test_prompt_still_covers_seniority_framing():
    prompt = _JOB_EXTRACTION_SYSTEM_PROMPT
    assert "Werkstudent" in prompt or "working student" in prompt
    assert "junior" in prompt.lower() and "senior" in prompt.lower()


def test_both_extraction_tools_share_the_same_prompt():
    """retrieve_web and retrieve_text must not drift into two different
    extraction behaviors for the same kind of posting."""
    import inspect

    from app.agents import job_agent

    web_source = inspect.getsource(job_agent.retrieve_web.func)
    text_source = inspect.getsource(job_agent.retrieve_text.func)
    assert "_JOB_EXTRACTION_SYSTEM_PROMPT" in web_source
    assert "_JOB_EXTRACTION_SYSTEM_PROMPT" in text_source
