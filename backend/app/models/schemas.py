"""
API Request and Response Models

This module defines all Pydantic models used for API communication.
These models provide validation, serialization, and documentation for the API.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal, Union


# ============================================
# CV Agent Models
# ============================================

class CVUploadRequest(BaseModel):
    """Request to extract CV information from a PDF file."""
    pdf_path: str = Field(
        ...,
        description="Path to the PDF resume file",
        example="C:/Documents/resume.pdf"
    )


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
    years_experience: Optional[int] = Field(default=None, description="Years of experience required")
    employment_type: str = Field(description="Employment type (Full-time, Contract, etc.)")
    location: str = Field(description="Job location")
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
    kind: Literal["direct", "family", "transferable", "missing"] = Field(
        description=(
            "How the match was resolved: direct (exact/alias), family "
            "(same skill-graph family), transferable (LLM/embedding judged "
            "transferable), or missing"
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


class DimensionScore(BaseModel):
    """Score for a single dimension of compatibility."""
    name: str = Field(description="Dimension name (hard_skills, experience, seniority, domain, ats_keywords)")
    score: float = Field(ge=0.0, le=1.0, description="Normalized 0-1 score for this dimension")
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this dimension in the aggregate")
    rationale: str = Field(default="", description="Short explanation of how the score was reached")


class GapAnalysis(BaseModel):
    """Bucketed view of where the candidate matches, can stretch, or falls short."""
    matched_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with transferability >= 0.95 (effectively direct matches)"
    )
    transferable_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with 0.55 <= transferability < 0.95 (each carries a bridge_bullet)"
    )
    missing_skills: List[SkillMatch] = Field(
        default_factory=list,
        description="Skills with transferability < 0.55 (real gaps)"
    )
    over_qualified_signals: List[str] = Field(
        default_factory=list,
        description="Senior signals on the CV not requested by the job (LLM-flagged)"
    )


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
    choice: Literal["accept", "keep_original", "custom", "include", "exclude"] = Field(
        description=(
            "Rewrite gate: 'accept' the rewrite, 'keep_original' text, or "
            "'custom' (see `text`). Selection gate: 'include' or 'exclude'."
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


