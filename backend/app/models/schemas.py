"""
API Request and Response Models

This module defines all Pydantic models used for API communication.
These models provide validation, serialization, and documentation for the API.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal, Union


# ============================================
# CV Agent Models
# ============================================

class ExperienceEntry(BaseModel):
    """One work-experience entry. OpenAI structured output requires explicit
    fields with ``extra='forbid'`` — ``Dict[str, Any]`` unions are rejected."""
    model_config = ConfigDict(extra="forbid")

    position: Optional[str] = Field(default=None, description="Job title or role")
    company: Optional[str] = Field(default=None, description="Employer name")
    location: Optional[str] = Field(default=None, description="City/country of the role, e.g. 'Aachen, Germany'")
    duration: Optional[str] = Field(default=None, description="Employment dates or duration")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Bullet points describing achievements in this role",
    )


class EducationEntry(BaseModel):
    """One education entry (one degree/program per entry)."""
    model_config = ConfigDict(extra="forbid")

    institution: Optional[str] = Field(default=None, description="School or university name")
    degree: Optional[str] = Field(
        default=None,
        description="Degree or program, e.g. 'M.Sc. in Computer Science' or 'Higher School Certificate'",
    )
    location: Optional[str] = Field(default=None, description="City/country, e.g. 'Aachen, Germany'")
    dates: Optional[str] = Field(
        default=None,
        description="Dates or graduation info as written, e.g. 'Expected: September 2027' or 'March 2026 -- August 2026'",
    )
    grade: Optional[str] = Field(
        default=None,
        description="Grade/GPA line if stated, e.g. '1.6 (German System)'",
    )
    details: List[str] = Field(
        default_factory=list,
        description="Extra lines such as awards, honors, or relevant coursework",
    )


class LeadershipEntry(BaseModel):
    """One leadership / extracurricular / volunteer activity."""
    model_config = ConfigDict(extra="forbid")

    role: Optional[str] = Field(default=None, description="Role or position title")
    organization: Optional[str] = Field(default=None, description="Organization or club name")
    location: Optional[str] = Field(default=None, description="City/country if stated")
    dates: Optional[str] = Field(default=None, description="Dates as written, e.g. 'October 2025 -- January 2026'")
    description: Optional[str] = Field(default=None, description="Short description of the activity")
    highlights: List[str] = Field(
        default_factory=list,
        description="Bullet points describing impact/responsibilities",
    )


class ProjectEntry(BaseModel):
    """One project entry."""
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, description="Project name")
    description: Optional[str] = Field(default=None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    outcomes: List[str] = Field(default_factory=list, description="Outcomes or achievements")


class LanguageEntry(BaseModel):
    """One spoken language with its stated proficiency."""
    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="Language name, e.g. 'German' or 'English'")
    fluency: Optional[str] = Field(
        default=None,
        description="Proficiency as written, e.g. 'native', 'C1', 'fluent', 'business fluent'",
    )


class ResumeInfo(BaseModel):
    """Structured resume information extracted from CV."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Full name of the applicant")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    location: Optional[str] = Field(
        default=None,
        description="Location (city, state, country, or full address)"
    )
    github_url: Optional[str] = Field(
        default=None,
        description="GitHub profile URL"
    )
    linkedin_url: Optional[str] = Field(
        default=None,
        description="LinkedIn profile URL"
    )
    portfolio_url: Optional[str] = Field(
        default=None,
        description="Personal portfolio or website URL"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Short professional summary / profile statement if present on the resume"
    )
    skills: List[str] = Field(description="List of professional skills")
    education: List[EducationEntry] = Field(
        description="Education entries with institution, degree, location, dates, and grade."
    )
    experience: List[ExperienceEntry] = Field(
        description="Work experience entries with position, company, duration, and responsibilities."
    )
    projects: List[ProjectEntry] = Field(
        description="Project entries with name, description, technologies, and outcomes."
    )
    leadership_activities: Optional[List[LeadershipEntry]] = Field(
        default=[],
        description="Leadership roles, extracurricular activities, volunteer work, or other relevant activities"
    )
    languages: List[LanguageEntry] = Field(
        default_factory=list,
        description="Spoken languages with proficiency, one entry per language (e.g. German native, English C1)",
    )


FactKind = Literal[
    "bullet", "skill", "language", "education", "employer", "title",
    "date_range", "metric", "certification", "user_stated",
]


class FactEntry(BaseModel):
    """
    One thing the candidate's CV (or the candidate, in this session) states.
    The fact ledger is the set of these; rewrites may only claim what some
    fact supports.
    """
    fact_id: str = Field(description=(
        "Stable id. Bullet facts reuse the bullet_id; others look like "
        "'skill.python', 'lang.german', 'experience.0.dates', 'experience.0.responsibilities.1.metric.0'."
    ))
    kind: FactKind
    value: str = Field(description="Canonical value, e.g. 'Python', 'German (native)', 'Jan 2022 - Aug 2024'.")
    source: Literal["cv_extraction", "user_confirmed", "user_added"]
    source_text: str = Field(description="The CV text it came from, or the user's own words.")
    bullet_id: Optional[str] = None
    entities: List[str] = Field(
        default_factory=list,
        description="Technologies named in this fact, as skill_graph.normalize() forms.",
    )
    user_verified: bool = Field(
        default=False,
        description="True when the user explicitly confirmed it in this session (an answer or a confirm gate).",
    )


class CVExtractionResponse(BaseModel):
    """Response from CV extraction."""
    success: bool
    cv_data: Optional[ResumeInfo] = None
    needs_clarification: bool = False
    questions: Optional[List[str]] = None
    message: str


class CVClarificationRequest(BaseModel):
    """Request to update CV with clarification answers."""
    cv_data: Dict[str, Any] = Field(description="Original CV data as JSON")
    clarifications: str = Field(description="User's answers to clarifying questions")


class CVClarificationResponse(BaseModel):
    """Response from CV clarification."""
    success: bool
    updated_cv_data: Optional[ResumeInfo] = None
    message: str


# ============================================
# Job Agent Models
# ============================================

class JobURLRequest(BaseModel):
    """Request to extract job information from URL."""
    urls: List[str] = Field(
        ...,
        description="List of job posting URLs",
        example=["https://example.com/careers/job/123"]
    )


class JobTextRequest(BaseModel):
    """Request to extract job information from pasted text."""
    job_text: str = Field(
        ...,
        description="Raw job posting text",
        example="We are looking for a Senior Software Engineer..."
    )


class JobRequirements(BaseModel):
    """Structured job requirements extracted from posting."""
    job_title: str = Field(description="Job title")
    short_title: str = Field(
        default="",
        description=(
            "The role in one to three words, as a person would say it: 'Software Developer', "
            "'Data Scientist'. No gender markers like (m/w/d), no seniority, no 'Student' or "
            "'Intern' prefix, no team or project name."
        ),
    )
    job_level: str = Field(description="Experience level (entry, mid, senior, etc.)")
    required_skills: List[str] = Field(description="Required technical skills")
    required_skill_groups: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Required skills grouped by OR-alternatives: each inner list means "
            "'at least one of these satisfies this requirement', not all of "
            "them. E.g. a posting asking for 'Java and WildFly/Quarkus/Spring "
            "Boot' produces two groups: ['Java'] and ['WildFly', 'Quarkus', "
            "'Spring Boot'] - a candidate with Java and Spring Boot fully "
            "satisfies both groups even without WildFly or Quarkus. A "
            "required skill with no stated alternative is still its own "
            "single-item group. `required_skills` remains the flattened "
            "union of every group's members, for display and backward "
            "compatibility; leave this empty only when the posting truly "
            "lists no OR-alternatives (every required skill is its own group "
            "of one, implicitly)."
        ),
    )
    preferred_skills: List[str] = Field(default=[], description="Preferred skills")
    learnable_skills: List[str] = Field(
        default_factory=list,
        description=(
            "Skills the posting says the employer will teach or that the candidate "
            "may still be learning, e.g. 'was Du noch nicht kennst, lernst Du bei "
            "uns', 'training provided', 'eager to learn X'. A skill listed here "
            "appears in neither required_skills nor preferred_skills."
        ),
    )
    required_languages: List[str] = Field(
        default_factory=list,
        description=(
            "Spoken languages the posting requires, with level when stated, e.g. "
            "['German (C1)', 'English (fluent)']. Never put languages into "
            "required_skills; they are eligibility requirements, not technical skills."
        ),
    )
    education_requirements: List[str] = Field(
        default_factory=list,
        description=(
            "Degree, field-of-study, or enrollment requirements, one per entry, e.g. "
            "['Enrolled student in Computer Science or related field', "
            "\"Bachelor's degree in Engineering\"]. Never put these into required_skills."
        ),
    )
    years_experience: Optional[int] = Field(default=None, description="Years of experience required")
    employment_type: str = Field(description="Employment type (Full-time, Contract, etc.)")
    location: str = Field(description="Job location")
    # Knockout conditions: stated hard conditions that rule a candidate out
    # regardless of skills. Each stays empty unless the posting states it.
    work_mode: Optional[Literal["onsite", "hybrid", "remote"]] = Field(
        default=None,
        description="Where the work happens, only when the posting says so.",
    )
    location_constraint: Optional[str] = Field(
        default=None,
        description=(
            "A stated condition on where the candidate must live or be able to "
            "work, e.g. 'within commuting distance of Munich', 'EU residents only'. "
            "Not the office address; that is `location`."
        ),
    )
    hours_per_week: Optional[str] = Field(
        default=None,
        description="Weekly hours as stated, e.g. '20', '16-20', 'up to 20'.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date as stated, e.g. 'immediately', 'ab sofort', '2026-11-01'.",
    )
    work_permit: Optional[str] = Field(
        default=None,
        description=(
            "A stated work-authorisation condition, e.g. 'EU work permit required', "
            "'no visa sponsorship'."
        ),
    )
    posting_language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 code of the language the posting is written in, e.g. 'de', 'en'.",
    )
    country: Optional[str] = Field(
        default=None,
        description="ISO 3166-1 alpha-2 code of the country the job is in, e.g. 'DE', 'AT', 'CH'.",
    )
    responsibilities: List[str] = Field(description="Key responsibilities")
    qualifications: List[str] = Field(default=[], description="Required qualifications")
    key_requirements: List[str] = Field(description="Critical must-have requirements")
    recipient_name: Optional[str] = Field(
        default=None,
        description=(
            "Named contact or hiring manager from the posting, if explicitly stated "
            "(e.g. 'Contact: Jane Smith'). Leave empty when not mentioned."
        ),
    )

    @field_validator("work_mode", mode="before")
    @classmethod
    def _normalise_work_mode(cls, value: Any) -> Optional[str]:
        # Models and older payloads write "On-site", "Remote-first",
        # "Homeoffice"; an unrecognised value must not fail the whole posting.
        if not isinstance(value, str):
            return None
        text = value.strip().lower()
        if "hybrid" in text:
            return "hybrid"
        if "remote" in text or "homeoffice" in text or "home office" in text:
            return "remote"
        if text.replace("-", "").replace(" ", "") in {"onsite", "office", "inoffice", "vorort"}:
            return "onsite"
        return None

    @field_validator("posting_language", mode="before")
    @classmethod
    def _normalise_language_code(cls, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip().lower()

    @field_validator("country", mode="before")
    @classmethod
    def _normalise_country_code(cls, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip().upper()


class JobExtractionResponse(BaseModel):
    """Response from job extraction."""
    success: bool
    job_data: Optional[JobRequirements] = None
    message: str


class CompanyResearchRequest(BaseModel):
    """Request to research company information."""
    company_name: str = Field(
        ...,
        description="Name of the company to research",
        example="Google"
    )


class CompanyInfo(BaseModel):
    """Structured company information from research."""
    company_name: str = Field(description="Official company name")
    industry: str = Field(description="Industry sector")
    company_size: Optional[str] = Field(default=None, description="Company size category")
    mission_statement: Optional[str] = Field(default=None, description="Mission statement")
    core_values: List[str] = Field(description="Core values and principles")
    recent_news: List[str] = Field(default=[], description="Recent news and developments")
    company_culture: str = Field(description="Company culture description")
    products_services: List[str] = Field(default=[], description="Main products or services")


class CompanyResearchResponse(BaseModel):
    """Response from company research."""
    success: bool
    company_data: Optional[CompanyInfo] = None
    message: str


# ============================================
# Writer Agent Models
# ============================================

class CVJobAlignmentRequest(BaseModel):
    """Request to analyze CV-job alignment."""
    cv_data: Dict[str, Any] = Field(description="Resume data from CV agent")
    job_data: Dict[str, Any] = Field(description="Job requirements from job agent")


class CVTailoringPlan(BaseModel):
    """Plan for tailoring CV to match job."""
    matching_experiences: List[str] = Field(description="Matching experience entries")
    matching_skills: List[str] = Field(description="Matching skills")
    relevant_projects: List[str] = Field(description="Relevant projects")
    keywords_to_incorporate: List[str] = Field(description="Keywords to include")
    reordering_suggestions: str = Field(description="Suggestions for reordering")
    emphasis_points: List[str] = Field(description="Points to emphasize")
    reasoning: str = Field(description="Reasoning for changes")


# ============================================
# Compatibility Scoring v2 Models
# ============================================

class SkillMatch(BaseModel):
    """A single required/preferred job skill resolved against the candidate's CV."""
    required_skill: str = Field(description="Skill name as it appears in the job posting")
    matched_with: Optional[str] = Field(
        default=None,
        description="Candidate skill that satisfies the requirement (None if missing)"
    )
    kind: Literal["direct", "family", "transferable", "learnable", "missing"] = Field(
        description=(
            "How the match was resolved: direct (exact/alias), family "
            "(same skill-graph family), transferable (LLM/embedding judged "
            "transferable), learnable (not held, but adjacent experience makes "
            "it learnable - a gap, not coverage), or missing"
        )
    )
    transferability: float = Field(
        ge=0.0,
        le=1.0,
        description="0-1 score for how well the candidate skill substitutes for the required one"
    )
    rationale: str = Field(
        default="",
        description="Short human-readable explanation of why this match was assigned"
    )
    bridge_bullet: Optional[str] = Field(
        default=None,
        description=(
            "Suggested resume bullet that highlights the bridge from the "
            "candidate's existing skill to the required one (only populated "
            "for transferable matches)"
        )
    )
    is_required: bool = Field(
        default=True,
        description="True if this skill came from required_skills, False if preferred"
    )
    scored_by: Optional[Literal["deterministic", "skill_graph", "embedding", "llm_judge"]] = Field(
        default=None,
        description="Which stage of the matching cascade decided this match.",
    )


class DimensionScore(BaseModel):
    """Score for a single dimension of compatibility."""
    name: str = Field(description="Dimension name (hard_skills, experience, seniority, domain, ats_keywords)")
    score: float = Field(ge=0.0, le=1.0, description="Normalized 0-1 score for this dimension")
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this dimension in the aggregate")
    rationale: str = Field(default="", description="Short explanation of how the score was reached")


class EligibilityCheck(BaseModel):
    """A non-technical must-have (language, degree, enrollment) checked against the CV."""
    requirement: str = Field(description="Requirement text as extracted from the posting")
    kind: Literal["language", "education"] = Field(description="What kind of requirement this is")
    status: Literal["met", "unclear", "unmet"] = Field(
        description=(
            "met: the CV satisfies it; unclear: the CV has no information to decide "
            "(e.g. no languages section); unmet: the CV has the information and it falls short"
        )
    )
    matched_with: Optional[str] = Field(default=None, description="CV entry that satisfied the requirement, if any")
    rationale: str = Field(default="", description="Short explanation of the status")


class GapAnalysis(BaseModel):
    """Bucketed view of where the candidate matches, can stretch, or falls short."""
    matched_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with transferability >= 0.95 (effectively direct matches)"
    )
    transferable_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with 0.55 <= transferability < 0.95, counted as covered via a related skill"
    )
    learnable_skills: List[SkillMatch] = Field(
        default_factory=list,
        description=(
            "Skills the CV does not show but adjacent experience makes learnable "
            "(kind='learnable'). A gap, not coverage."
        ),
    )
    missing_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with transferability < 0.55 and no adjacent experience (real gaps)"
    )
    over_qualified_signals: List[str] = Field(
        default_factory=list,
        description="Senior signals on the CV not requested by the job (LLM-flagged)"
    )
    eligibility: List[EligibilityCheck] = Field(
        default_factory=list,
        description="Language, degree, and enrollment requirements checked against the CV",
    )


RequirementStatus = Literal["met", "partial", "learnable_gap", "missing", "unclear"]


class RequirementRow(BaseModel):
    """One requirement from the posting, how the CV meets it, and where it shows."""
    requirement_id: str = Field(description="'req.<n>' for skills, 'ko.<n>' for knockouts; stable for a posting.")
    text: str = Field(description="The requirement as written in the posting.")
    category: Literal["must_have", "nice_to_have", "learnable", "knockout"]
    alternatives: List[str] = Field(
        default_factory=list, description="OR-group members when any one of them satisfies it.",
    )
    status: RequirementStatus
    matched_with: Optional[str] = Field(
        default=None, description="The CV skill that meets it, or the adjacent one for a learnable gap.",
    )
    evidence_fact_ids: List[str] = Field(
        default_factory=list, description="Fact-ledger ids that show the requirement (skills, bullets, languages).",
    )
    evidence_bullet_ids: List[str] = Field(
        default_factory=list, description="The CV bullets among them: where a recruiter would see it.",
    )
    reason: str = ""
    scored_by: Literal["deterministic", "skill_graph", "embedding", "llm_judge"] = "deterministic"
    judge_model: Optional[str] = Field(default=None, description="Set when an LLM judge decided the status.")


class KeywordPlanItem(BaseModel):
    """A posting term the tailored CV may use, and the facts that justify it."""
    keyword: str = Field(description="The term as the posting writes it, e.g. 'REST APIs'.")
    requirement_id: str = Field(default="", description="The rubric row it serves, e.g. 'req.0'.")
    allowed: bool = Field(
        default=False,
        description="Set by validation, not trusted from the model: True only when a cited fact names the term.",
    )
    evidence_fact_ids: List[str] = Field(
        default_factory=list, description="Fact-list ids that show the candidate has it.",
    )


class SkillGroup(BaseModel):
    """One labelled line of the CV's Skills section."""
    label: str = Field(description="Group label in the CV's language, e.g. 'Programming languages'.")
    skills: List[str] = Field(default_factory=list, description="Skills exactly as the CV lists them.")


class StrategistPlan(BaseModel):
    """The holistic tailoring plan, reviewed together with the score."""
    positioning: str = Field(description=(
        "One short paragraph: how to present this candidate for this role, honestly, "
        "built on what the CV shows."
    ))
    headline: Optional[str] = Field(default=None, description=(
        "A one-line profile headline for the top of the CV, using only facts from the list, "
        "written in the language the CV is currently in; null when nothing honest fits."
    ))
    skill_groups: List["SkillGroup"] = Field(default_factory=list, description=(
        "The CV's skills in 3-4 labelled groups (e.g. programming languages, backend/frontend, "
        "ML/agents, tools), most relevant group first. Only skills from the CV's skills list."
    ))
    cv_language: Literal["english", "german"] = Field(description="Language the tailored CV should be in.")
    convention: Literal["international", "dach"] = Field(description=(
        "'dach' for postings in Germany, Austria or Switzerland, otherwise 'international'."
    ))
    section_order: List[str] = Field(default_factory=list, description=(
        "CV sections, most important first, from: experience, projects, education, skills, "
        "leadership_activities, languages, certificates, awards, publications, interests."
    ))
    project_order: List[int] = Field(default_factory=list, description="Indices into the CV's projects, best first.")
    bullets_to_feature: List[str] = Field(default_factory=list, description=(
        "bullet_ids from the bullet list that best evidence the must-haves."
    ))
    keyword_plan: List[KeywordPlanItem] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list, description=(
        "What a recruiter may question (unmet must-haves, unclear knockouts, gaps), "
        "based only on the report and the CV."
    ))


class CompatibilityReport(BaseModel):
    """Multi-dimensional compatibility output produced by calculate_compatibility_score_v2."""
    aggregate_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted aggregate of all dimension scores (0-1)"
    )
    level: Literal["low", "medium", "high", "excellent", "unknown"] = Field(
        description="Discrete band derived from aggregate_score"
    )
    interpretation: str = Field(
        default="",
        description="Short human-readable summary of the report"
    )
    dimensions: List[DimensionScore] = Field(
        default_factory=list,
        description="Per-dimension scores and rationales"
    )
    gap_analysis: GapAnalysis = Field(
        default_factory=GapAnalysis,
        description="Skill bucketing and over-qualification signals"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while computing the report"
    )
    rubric: List[RequirementRow] = Field(
        default_factory=list,
        description="Every skill requirement with its status and CV evidence.",
    )
    knockouts: List[RequirementRow] = Field(
        default_factory=list,
        description=(
            "Pass/fail conditions (languages, education, work mode, location, hours, "
            "start date, work permit). Not part of aggregate_score; an unmet one caps the level."
        ),
    )


class JobSummary(BaseModel):
    """Structured summary of a job posting."""
    role: str = Field(description="Job title and seniority level")
    responsibilities: List[str] = Field(description="Key responsibilities and duties")
    required_skills: List[str] = Field(description="Technical and soft skills required by the posting")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills mentioned in the posting")
    key_notes: List[str] = Field(default_factory=list, description="Other important points (culture fit, location, employment type, etc.)")


class SelectedBullet(BaseModel):
    """A single resume bullet selected for emphasis."""
    section: str = Field(description="Section name this bullet belongs to (e.g. 'experience', 'projects')")
    original_text: str = Field(description="Original bullet text from the CV")
    relevance_score: float = Field(ge=0.0, le=1.0, description="0-1 relevance score to the target job")
    reason: str = Field(default="", description="Why this bullet was selected")
    bullet_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable id of this bullet from the CV bullet index (e.g. "
            "'experience.0.responsibilities.2'). Echoed verbatim from the "
            "numbered bullet list given in the prompt; None if the model "
            "could not match an id."
        ),
    )


class SelectedContent(BaseModel):
    """Output of the content selection and prioritization step."""
    selected_bullets: List[SelectedBullet] = Field(
        default_factory=list,
        description="Top bullets per section ordered by relevance"
    )
    section_order: List[str] = Field(
        default_factory=list,
        description="Recommended section ordering for the tailored CV (most impactful first)"
    )
    sections_to_emphasize: List[str] = Field(
        default_factory=list,
        description="Section names that should receive the most visual weight"
    )
    items_to_de_emphasize: List[str] = Field(
        default_factory=list,
        description="Bullet texts or section names to downplay or remove"
    )


class BulletRewrite(BaseModel):
    """A single bullet rewrite."""
    original: str = Field(description="Original bullet text")
    rewritten: str = Field(description="Enhanced bullet text (no fabrication)")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 score indicating how well the rewritten bullet matches the job")
    keywords_added: List[str] = Field(default_factory=list, description="Job keywords naturally woven into the rewrite")
    bullet_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable id of the bullet being rewritten, echoed verbatim from "
            "the numbered bullet list given in the prompt; None if the model "
            "could not match an id."
        ),
    )
    cited_fact_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Ids from the fact list that support anything the rewrite adds beyond "
            "the original bullet's own wording (e.g. a skill from the skills list). "
            "Empty when it only rewords the original."
        ),
    )


class VerifierIssue(BaseModel):
    """One problem the rewrite verifier found."""
    bullet_id: Optional[str] = None
    check: Literal[
        "unknown_entity", "new_metric", "changed_dates", "missing_citation",
        "banned_phrase", "length_blowup", "language_mismatch", "structure_changed",
    ]
    severity: Literal["block", "warn"]
    detail: str
    offending_span: Optional[str] = None


class VerifierResult(BaseModel):
    """Outcome of checking one rewrite pass against the fact ledger."""
    passed: bool = Field(description="No 'block' issues.")
    attempt: int = Field(ge=1)
    issues: List[VerifierIssue] = Field(default_factory=list)
    preservation_score: float = Field(
        ge=0.0, le=1.0,
        description="Share of the originals' technologies and numbers the rewrites kept.",
    )

    def blocked_ids(self) -> List[str]:
        return list(dict.fromkeys(i.bullet_id for i in self.issues if i.severity == "block" and i.bullet_id))


class RewrittenContent(BaseModel):
    """Output of the content rewrite and enhancement step."""
    rewritten_bullets: List[BulletRewrite] = Field(
        default_factory=list,
        description="Enhanced bullet points with original and rewritten versions"
    )
    updated_summary: str = Field(
        default="",
        description="Enhanced professional summary / headline (empty string if not applicable)"
    )
    keywords_inserted: List[str] = Field(
        default_factory=list,
        description="Complete list of job keywords incorporated across all rewrites"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues, e.g. rewrites that could not be matched to a CV bullet"
    )


class CVJobAlignmentResponse(BaseModel):
    """Response from alignment analysis."""
    success: bool
    tailoring_plan: Optional[CVTailoringPlan] = None
    message: str


class GenerateTailoredCVRequest(BaseModel):
    """Request to generate tailored CV."""
    cv_data: Dict[str, Any] = Field(description="Original CV data")
    tailoring_plan: Dict[str, Any] = Field(description="Tailoring plan from alignment analysis")
    output_filename: str = Field(
        description="Desired filename for PDF",
        example="john_doe_cv_tailored.pdf"
    )


class GenerateTailoredCVResponse(BaseModel):
    """Response from CV generation."""
    success: bool
    pdf_path: Optional[str] = None
    latex_preview: Optional[str] = None
    message: str


class GenerateCoverLetterRequest(BaseModel):
    """Request to generate cover letter."""
    cv_data: Dict[str, Any] = Field(description="Resume data")
    job_data: Dict[str, Any] = Field(description="Job requirements")
    company_data: Optional[Dict[str, Any]] = Field(default=None, description="Optional company info")
    output_filename: str = Field(
        description="Desired filename for PDF",
        example="john_doe_cover_letter.pdf"
    )
    recipient_info: Optional[str] = Field(
        default=None,
        description=(
            "Who the letter is addressed to. Required unless recipient_name "
            "was extracted from the job posting."
        ),
    )


class CoverLetterContent(BaseModel):
    """Structured cover letter content.

    For English letters the paragraph fields map as:
      opening_paragraph  → hook / interest
      body_paragraph_1/2 → experience ↔ job alignment
      body_paragraph_3   → optional company-specific point
      closing_paragraph  → call to action

    For German Anschreiben the same fields map as:
      opening_paragraph  → Einleitung
      body_paragraph_1/2 → Hauptteil
      body_paragraph_3   → optional second Hauptteil paragraph
      closing_paragraph  → Schlussteil
    Additionally betreff and grussformel are populated.
    """
    language: str = Field(
        default="english",
        description="Output language/format: 'english' for a standard English cover letter, 'german' for a formal German Anschreiben (Sie-form)"
    )
    # --- German-only fields (empty string for English) ---
    betreff: str = Field(
        default="",
        description="German Betreff (subject line), e.g. 'Bewerbung als Senior Software Engineer'. Populated only for language='german'."
    )
    grussformel: str = Field(
        default="",
        description="German Grußformel (valediction), e.g. 'Mit freundlichen Grüßen'. Populated only for language='german'."
    )
    # --- Paragraph content (used by both English and German) ---
    opening_paragraph: str = Field(description="English: opening hook. German: Einleitung paragraph.")
    body_paragraph_1: str = Field(description="English: first experience paragraph. German: first Hauptteil paragraph.")
    body_paragraph_2: str = Field(description="English: second qualifications paragraph. German: second Hauptteil paragraph.")
    body_paragraph_3: str = Field(default="", description="Optional. English: company-specific point. German: optional additional Hauptteil paragraph.")
    closing_paragraph: str = Field(description="English: call to action / appreciation. German: Schlussteil paragraph.")


class GenerateCoverLetterResponse(BaseModel):
    """Response from cover letter generation."""
    success: bool
    pdf_path: Optional[str] = None
    content: Optional[CoverLetterContent] = None
    message: str


# ============================================
# Shared file-output model
# ============================================

class GeneratedFile(BaseModel):
    """Metadata for a generated file."""
    filename: str = Field(description="Name of the generated file")
    file_type: str = Field(description="Type of file (cv, cover_letter, etc.)")
    download_url: str = Field(description="URL to download the file")


# ============================================
# Orchestrator API Models (LangGraph-driven flow)
# ============================================

class OrchestratorGatePayload(BaseModel):
    """Public shape of a pending human-in-the-loop gate."""
    step: str = Field(description="Stable gate identifier")
    kind: Literal["approval", "choice", "input"] = Field(description="Gate kind")
    narration: str = Field(description="Assistant text for this gate")
    preview: Dict[str, Any] = Field(default_factory=dict, description="Structured artifact under review")
    allowed_actions: List[Literal["approve", "reject", "edit", "choose"]] = Field(
        description="Actions the client may submit back"
    )
    choices: Optional[List[str]] = Field(
        default=None,
        description="Valid values for action='choose'. Populated only when kind='choice'."
    )


class BulletDecision(BaseModel):
    """Per-bullet decision attached to a gate resolution's `decisions` map."""
    # Same values as app.agents.orchestrator.gates.BulletDecisionChoice; a
    # value missing here is refused with 422 before the gate ever sees it.
    choice: Literal["accept", "keep_original", "custom", "include", "exclude", "leave_out"] = Field(
        description=(
            "Rewrite gate: 'accept' the rewrite, 'keep_original' text, or "
            "'custom' (see `text`). Selection gate: 'include' (tailor it), "
            "'exclude' (keep it as it is) or 'leave_out' (drop it from the "
            "tailored CV)."
        )
    )
    text: Optional[str] = Field(
        default=None,
        description="Replacement text when choice='custom'; ignored otherwise.",
    )


class OrchestratorGateResolution(BaseModel):
    """Client reply to a pending OrchestratorGatePayload."""
    action: Literal["approve", "reject", "edit", "choose"] = Field(
        description="Selected action. Must be in the gate's allowed_actions."
    )
    step: Optional[str] = Field(
        default=None,
        description=(
            "The `step` of the gate this answers. When set and it does not match "
            "the pending gate, the answer is stale and is refused with 409."
        ),
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Free-text edit instructions (required when action='edit')."
    )
    choice: Optional[str] = Field(
        default=None,
        description="Selected option for kind='choice' gates (required when action='choose')."
    )
    decisions: Optional[Dict[str, BulletDecision]] = Field(
        default=None,
        description=(
            "Per-bullet decisions keyed by bullet_id, for approve_selection "
            "and approve_rewrite gates. Bullets with no entry default to "
            "'accept'/'include'."
        ),
    )
    answers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Answers keyed by question id, for gates whose preview lists `questions` (confirm_eligibility).",
    )


class OrchestratorStartRequest(BaseModel):
    """Initialise a new orchestrator session and run until the first gate."""
    flow: Literal["job_tailoring", "cv_review", "discovery"] = Field(
        description="Which entry branch of the orchestrator graph to enter."
    )
    cv_data: Dict[str, Any] = Field(description="Resume data from CV agent (ResumeInfo dict)")
    job_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Job requirements from job agent (required for job_tailoring)."
    )
    company_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional company info from company research."
    )
    cv_language: Optional[Literal["english", "german"]] = Field(
        default=None,
        description=(
            "The CV language, when the user already chose it. Without it, the "
            "flow asks after scoring (job_tailoring only)."
        ),
    )


class OrchestratorMessageRequest(BaseModel):
    """Send either a free-text chat turn or a structured gate resolution."""
    session_id: str = Field(description="Session identifier from /start")
    kind: Literal["chat", "gate_resolution"] = Field(
        description="'gate_resolution' resumes the graph from its interrupt; 'chat' is reserved for free-text turns outside any gate."
    )
    text: Optional[str] = Field(
        default=None,
        description="Free-text message (required when kind='chat')."
    )
    resolution: Optional[OrchestratorGateResolution] = Field(
        default=None,
        description="Structured gate reply (required when kind='gate_resolution')."
    )


class OrchestratorResponse(BaseModel):
    """Response from /start and /message - one shape for both."""
    success: bool
    session_id: str
    narration: str = Field(default="", description="Latest assistant text from the graph.")
    narrations: List[str] = Field(
        default_factory=list,
        description=(
            "Every assistant narration produced since the previous response, "
            "in order. The frontend renders one chat bubble per entry. "
            "`narration` mirrors the last relevant entry for older clients."
        ),
    )
    pending_gate: Optional[OrchestratorGatePayload] = Field(
        default=None,
        description="Populated when the graph is paused on a gate; null when the run is complete."
    )
    generated_files: List[GeneratedFile] = Field(
        default_factory=list,
        description="Files produced by the run so far."
    )
    done: bool = Field(
        default=False,
        description="True when the graph has reached END."
    )
    message: str = Field(default="ok", description="Status message")
    run_id: Optional[str] = Field(
        default=None,
        description=(
            "Id of the graph run this request started, for finding its trace. "
            "Null when the request answered without running the graph."
        ),
    )


class OrchestratorStateResponse(BaseModel):
    """Debug snapshot of the current state of a session."""
    success: bool
    session_id: str
    flow: Optional[str] = None
    pending_gate: Optional[OrchestratorGatePayload] = None
    generated_files: List[GeneratedFile] = Field(default_factory=list)
    last_narration: str = ""
    done: bool = False


# ============================================
# Documents API Models (the finished CV and cover letter in the editor)
# ============================================

class CvEdit(BaseModel):
    """One change to the finished CV, addressed by a path from the CV's view."""
    op: Literal["set", "insert", "delete"]
    path: str = Field(..., min_length=1, max_length=200, description=(
        "set: a segment's src path (e.g. 'experience.0.company'); insert: a bullet list's "
        "bullets_src path; delete: a bullet's path (the list path plus its index)."
    ))
    value: Optional[Union[str, List[str]]] = Field(default=None, description="The new text (set, insert).")
    index: Optional[int] = Field(default=None, ge=0, description="insert: the position; the end when omitted.")


class CvEditRequest(BaseModel):
    """Edits to the CV, applied in order to the revision the editor was showing."""
    revision: int = Field(..., ge=0, description="The document revision the edits were made on.")
    edits: List[CvEdit] = Field(default_factory=list, max_length=500)


class CoverLetterEditRequest(BaseModel):
    """New values for the letter's editable fields (the ones in CoverLetterView.fields)."""
    revision: int = Field(..., ge=0, description="The document revision the edits were made on.")
    fields: Dict[str, str] = Field(default_factory=dict)


class DocumentResetRequest(BaseModel):
    """Go back to the generated version of a document."""
    revision: int = Field(..., ge=0, description="The document revision the editor was showing.")


class CvDocumentView(BaseModel):
    """The finished CV as the PDF prints it, with every printed piece's source field."""
    revision: int
    rendered_revision: Optional[int] = Field(
        default=None, description="The revision the PDF shows; differs from revision when generating it failed."
    )
    edited: bool = Field(description="True when the CV differs from the version the flow generated.")
    filename: str
    download_url: Optional[str] = None
    language: str
    header: Dict[str, Any] = Field(description="name / label / summary / contact_parts, each with its *_segs.")
    sections: List[Dict[str, Any]] = Field(description="{type, title, style, entries} with segments kept.")


class CoverLetterView(BaseModel):
    """The finished cover letter as the PDF prints it, and its editable fields."""
    revision: int
    rendered_revision: Optional[int] = None
    edited: bool
    filename: str
    download_url: Optional[str] = None
    language: str
    fields: Dict[str, str] = Field(description=(
        "The editable values: the paragraph fields, recipient, sign_off, and betreff for a German letter."
    ))
    letter: Dict[str, Any] = Field(description=(
        "Everything the PDF prints: header (name, phone, email, links), document_title, date_label, "
        "current_date, betreff, salutation, paragraph_fields [{field, text}], sign_off."
    ))


class DocumentsResponse(BaseModel):
    """The session's finished documents; null where the PDF doesn't exist yet."""
    success: bool = True
    session_id: str
    cv: Optional[CvDocumentView] = None
    cover_letter: Optional[CoverLetterView] = None
    editable: bool = Field(description=(
        "False while a proposed change waits for confirmation in the chat; saving is refused then."
    ))


class DocumentSaveResponse(BaseModel):
    """After a save or reset: the session as /message returns it, and the documents."""
    session: OrchestratorResponse
    documents: DocumentsResponse


# ============================================
# File Upload Models (for future multipart support)
# ============================================

class FileUploadResponse(BaseModel):
    """Response from file upload."""
    success: bool
    file_path: str
    filename: str
    message: str


# ============================================
# Audio Transcription Models
# ============================================

class TranscriptionRequest(BaseModel):
    """Request parameters for audio transcription (used for query params)."""
    model: Literal["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize"] = Field(
        default="gpt-4o-transcribe",
        description="Model to use for transcription"
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt", "diarized_json"] = Field(
        default="json",
        description="Format of the transcription response"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (ISO 639-1 or 639-3) for the audio"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text to guide the model's style or continue a previous audio segment"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0-1). Higher values make output more random."
    )
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(
        default=None,
        description="Granularity of timestamps (only for whisper-1)"
    )
    chunking_strategy: Optional[Literal["auto"]] = Field(
        default=None,
        description="Chunking strategy for long audio (required for gpt-4o-transcribe-diarize when audio > 30s)"
    )


class TranscriptionResponse(BaseModel):
    """Response from audio transcription."""
    success: bool
    text: Optional[str] = Field(default=None, description="Transcribed text")
    segments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Segments with timestamps (for verbose_json or diarized_json formats)"
    )
    words: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word-level timestamps (for verbose_json with word granularity)"
    )
    message: str


class TranslationRequest(BaseModel):
    """Request parameters for audio translation (used for query params)."""
    model: Literal["whisper-1"] = Field(
        default="whisper-1",
        description="Model to use for translation (only whisper-1 supported)"
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Field(
        default="json",
        description="Format of the translation response"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text to guide the model's style"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0-1)"
    )


class TranslationResponse(BaseModel):
    """Response from audio translation."""
    success: bool
    text: Optional[str] = Field(default=None, description="Translated text (always in English)")
    segments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Segments with timestamps (for verbose_json format)"
    )
    words: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Word-level timestamps (for verbose_json with word granularity)"
    )
    message: str


# ============================================
# Generic Response Models
# ============================================

class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    agents_available: List[str]
    message: str


