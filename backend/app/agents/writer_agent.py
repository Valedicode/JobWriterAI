"""
Writer Agent - CV and Cover Letter Tailoring Specialist

This agent receives structured data from CV and Job agents and generates:
1. Tailored CV PDFs based on job requirements
2. Cover letter PDFs aligned with the job and company

Architecture:
- Input: Pre-processed JSON data from cv_agent (ResumeInfo) and job_agent (JobRequirements, CompanyInfo)
- Processing: Gap analysis → Tailoring plan → Content generation → LaTeX → PDF
- Output: Professional PDF documents (CV and cover letter)
- Pattern: LangChain @tool functions invoked by the orchestrator graph

Key Features:
- Human-in-the-loop at every critical step (orchestrator gates)
- Preserves candidate's authentic voice
- Never fabricates content - only emphasizes/refines existing material
- LaTeX-based PDF generation via Jinja2 templates and Tectonic/pdflatex
- Company-aware cover letter generation
"""

# Note: `from langchain.agents import create_agent` was removed during the
# migration to the orchestrator graph. The legacy WriterAgent (a create_agent
# instance + giant system prompt) has been retired; the orchestrator now drives
# sequencing explicitly via graph nodes that invoke the @tool functions in
# this module directly.
from langchain_core.tools import tool
from app.openai_llm import chat_model
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import json
import dotenv
import os
import re
from typing import List, Dict, Any, Optional
from app.models.schemas import (
    CVTailoringPlan,
    CoverLetterContent,
    CompatibilityReport,
    JobSummary,
    SelectedContent,
    RewrittenContent,
    BulletRewrite,
)
from app.agents.scoring_agent import (
    calculate_semantic_similarity,
    calculate_bm25_score,
    calculate_compatibility_score,
    match_skill_pairs,
    assess_transferability_llm,
    calculate_compatibility_score_v2,
    build_gap_analysis,
    _calculate_compatibility_score_v2_internal,
)
from app.services.latex_renderer import (
    compile_tex_to_pdf,
    render_cover_letter_tex,
    render_cv_pdf,
    render_cv_tex,
)

dotenv.load_dotenv()

# ============================================
# Constants
# ============================================

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"


def _resolve_output_dir(output_dir: str) -> Path:
    """The requested output directory, which must lie inside data/ (normally
    a session folder from app.services.file_store)."""
    if not output_dir:
        return DEFAULT_OUTPUT_DIR
    root = DEFAULT_OUTPUT_DIR.resolve()
    target = Path(output_dir).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError("output_dir must be inside the data directory")
    return target

# ============================================
# Tools (Agent Capabilities)
# ============================================

@tool
def analyze_cv_job_alignment(cv_json: str, job_json: str) -> str:
    """
    Analyze how well the CV matches job requirements and create a tailoring plan.
    
    This tool performs gap analysis between the candidate's CV and the target job,
    identifying strengths to emphasize and areas where the CV can be optimized.
    
    IMPORTANT: This tool receives pre-processed JSON from cv_agent and job_agent.
    No extraction is needed - just analysis of structured data.
    
    Args:
        cv_json: ResumeInfo JSON string from cv_agent output
        job_json: JobRequirements JSON string from job_agent output
        
    Returns:
        JSON string with CVTailoringPlan containing specific, actionable suggestions
        
    Example:
        >>> plan = analyze_cv_job_alignment(cv_data, job_data)
        >>> # Returns detailed plan with matching experiences, skills, keywords, etc.
    """
    llm = chat_model("extract", temperature=0)
    structured_llm = llm.with_structured_output(CVTailoringPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert career counselor conducting gap analysis.
            Your task is to compare the candidate's CV with job requirements and identify:

            1. MATCHING EXPERIENCES: Which work experiences or roles align with job responsibilities?
            2. MATCHING SKILLS: Which technical and soft skills overlap between CV and job?
            3. RELEVANT PROJECTS: Which projects demonstrate capabilities needed for this job?
            4. KEYWORDS: What job-specific terms should appear in the tailored CV?
            5. REORDERING: Should sections be reorganized to put most relevant items first?
            6. EMPHASIS: What achievements deserve stronger highlighting?

            CRITICAL CONSTRAINTS:
            - Only work with EXISTING content from the CV
            - Never suggest fabricating experience or skills
            - Focus on emphasizing and reframing, not inventing
            - Be specific with actionable suggestions
            Provide concrete, implementable recommendations.
        """),
        ("user", """
        Candidate's CV:
        {cv}

        Job Requirements:
        {job}

        Analyze the alignment and create a detailed tailoring plan.
        """)
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"cv": cv_json, "job": job_json})
    
    return result.model_dump_json(indent=2)

@tool
def generate_cover_letter_content(cv_json: str, job_json: str, company_json: str = "", language: str = "english") -> str:
    """
    Generate tailored cover letter content for the job application.

    Creates a compelling, authentic cover letter that connects the candidate's
    background to the job requirements and company culture.

    Args:
        cv_json: ResumeInfo JSON from cv_agent
        job_json: JobRequirements JSON from job_agent
        company_json: Optional CompanyInfo JSON from job_agent for company-specific points
        language: Output language/format.
            "english" (default) → standard English cover letter (3-4 paragraphs)
            "german"            → formal German Anschreiben in Sie-form with
                                  Betreff, Einleitung, Hauptteil, Schlussteil,
                                  and Grußformel

    Returns:
        JSON string with CoverLetterContent structure, or {"error": ...}
        when the call fails.
        For English: opening/body/closing paragraphs populated; betreff and
        grussformel are empty strings.
        For German: betreff and grussformel populated; opening_paragraph →
        Einleitung, body_paragraph_1/2 → Hauptteil, closing_paragraph →
        Schlussteil; language field set to "german".

    Note:
        English letter structure (3-4 paragraphs):
        - Opening: Express interest and how you learned about the role
        - Body 1-2: Connect relevant experiences to job requirements
        - Body 3 (optional): Address company-specific points
        - Closing: Call to action and appreciation

        German Anschreiben structure:
        - Betreff: Subject line naming the exact position
        - Anrede: Formal salutation (generated separately as recipient_info)
        - Einleitung: Why applying, brief self-introduction
        - Hauptteil (1-2 paragraphs): Specific qualifications and experience
        - Schlussteil: Motivation summary, request for interview
        - Grußformel: Formal valediction
    """
    llm = chat_model("writer", temperature=0.4)
    structured_llm = llm.with_structured_output(CoverLetterContent)

    # Build context with optional company info
    company_context = ""
    if company_json and company_json.strip():
        company_context = f"\n\nCompany Information:\n{company_json}"

    if language.lower() == "german":
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Du schreibst ein professionelles Anschreiben auf Deutsch für eine Bewerbung.

        FORMALE VORGABEN:
        1. Sprache: Ausschließlich Deutsch, formelles Sie durchgehend
        2. Register: Formell-sachlich, klar und präzise — kein Marketingjargon
        3. Länge: 250–350 Wörter Gesamttext (ohne Betreff und Grußformel)
        4. Aufbau:
           - betreff: Einzeiliger Betreff, der die exakte Stellenbezeichnung nennt,
             z. B. „Bewerbung als Senior Software Engineer (Ref.-Nr. 12345)"
           - opening_paragraph (Einleitung): Ein kurzer Absatz — warum diese Stelle,
             kurze Selbstvorstellung
           - body_paragraph_1 (1. Hauptteil): Konkrete Qualifikationen und
             Erfahrungen, die direkt auf die Stellenanforderungen einzahlen
           - body_paragraph_2 (2. Hauptteil): Weitere relevante Kompetenzen oder
             Projekte; bei Bedarf Unternehmenbezug
           - body_paragraph_3 (optional): Nur befüllen, wenn Unternehmensinformationen
             vorliegen und ein spezifischer Mehrwert hergestellt werden kann
           - closing_paragraph (Schlussteil): Motivation bekräftigen,
             Gesprächswunsch äußern, Verfügbarkeit nennen
           - grussformel: „Mit freundlichen Grüßen" (Standard) oder eine
             situationsangemessene Alternative

        5. Inhaltliche Prinzipien:
           - KONKRET: Echte Erfahrungen und Kenntnisse aus dem Lebenslauf referenzieren
           - AUSGERICHTET: Zeigen, dass die Anforderungen der Stelle verstanden wurden
           - MEHRWERT: Erklären, welchen Beitrag die Kandidatin/der Kandidat leistet
           - AUTHENTISCH: Den Stil der Bewerberin/des Bewerbers wahren

        VERMEIDEN:
        - Generische Floskeln wie „Hiermit bewerbe ich mich ..."
        - Reine Wiederholung des Lebenslaufs
        - Übertriebenes Eigenlob oder unterwürfige Formulierungen
        - Anglizismen und Buzzwords
        - Du-Form oder Kumpelton
        - Englische Wörter oder Satzteile"""),
            ("user", """Lebenslauf der Kandidatin / des Kandidaten:
        {cv}

        Stellenanforderungen:
        {job}
        {company}

        Schreibe ein überzeugendes, maßgeschneidertes Anschreiben auf Deutsch (Sie-Form).
        Setze language=\"german\" und befülle betreff sowie grussformel entsprechend.""")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are writing a professional cover letter.

        WRITING GUIDELINES:
        1. Tone: Professional yet personable, enthusiastic but not over-the-top
        2. Length: 300-400 words total (concise and impactful)
        3. Structure:
        - Opening: Hook with genuine interest in role/company
        - Body: Connect 2-3 most relevant experiences to job needs
        - Closing: Strong call to action

        4. Content principles:
        - Be SPECIFIC: Reference actual experiences and skills from CV
        - Show ALIGNMENT: Demonstrate understanding of job requirements
        - Add VALUE: Explain what you'll bring to the role
        - Be AUTHENTIC: Maintain candidate's voice

        5. If company info provided:
        - Reference company values or culture
        - Mention relevant company news or initiatives
        - Show you've done research

        AVOID:
        - Generic statements that could apply to any job
        - Simply repeating CV content
        - Explaining why YOU want the job (focus on what YOU bring)
        - Desperation or begging tone
        - Clichés and buzzwords"""),
            ("user", """Candidate's CV:
        {cv}

        Job Requirements:
        {job}
        {company}

        Write a compelling, tailored cover letter.""")
        ])

    try:
        chain = prompt | structured_llm
        result = chain.invoke({"cv": cv_json, "job": job_json, "company": company_context})
    except Exception as e:  # noqa: BLE001 - reported to the node as a step failure
        return json.dumps({"error": str(e)})

    return result.model_dump_json(indent=2)

@tool
def generate_cv_pdf(
    cv_json: str,
    output_filename: str,
    applicant_name: str,
    section_order_json: str = "",
    output_dir: str = "",
    convention: str = "international",
) -> str:
    """
    Generate a tailored CV PDF via LaTeX (Harvard-style, JSON Resume taxonomy).

    WARNING: Only call this tool after the user has explicitly approved the CV content!

    Args:
        cv_json: ResumeInfo or JSON Resume JSON (merged tailored CV data)
        output_filename: Filename for the PDF (e.g., "kevin_ha_cv_tailored.pdf")
        applicant_name: Full name for document metadata
        section_order_json: Optional JSON list of section keys for ordering
        output_dir: Directory to write into (a session folder inside data/);
            defaults to data/ itself
        convention: "dach" for DACH date style, else "international"

    Returns:
        Success message with the filename for download, then one
        "Layout note: ..." line per page-layout problem that couldn't be fixed
    """
    try:
        if not output_filename.lower().endswith(".pdf"):
            return "Error: Output filename must end with .pdf"

        try:
            cv_data = json.loads(cv_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid CV JSON data: {str(e)}"

        section_order: list[str] | None = None
        if section_order_json and section_order_json.strip():
            try:
                parsed = json.loads(section_order_json)
                if isinstance(parsed, list):
                    section_order = [s for s in parsed if isinstance(s, str)]
            except json.JSONDecodeError:
                pass

        target_dir = _resolve_output_dir(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(output_filename).stem
        pdf_path, layout = render_cv_pdf(
            cv_data, target_dir, stem, section_order=section_order,
            convention="dach" if convention == "dach" else "international",
        )
        final_path = target_dir / output_filename
        if pdf_path != final_path:
            pdf_path.replace(final_path)

        message = f"CV PDF generated successfully! The file '{output_filename}' is ready for download."
        notes = [f"{problem}; consider shortening or removing a section." for problem in layout["problems"]]
        return message + "".join(f"\nLayout note: {note}" for note in notes)

    except RuntimeError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error generating CV PDF: {str(e)}"


@tool
def generate_cover_letter_pdf(
    content_json: str,
    output_filename: str,
    applicant_name: str,
    applicant_contact: str = "",
    recipient_info: str = "",
    cv_json: str = "",
    job_json: str = "",
    output_dir: str = "",
) -> str:
    """
    Generate cover letter PDF from structured content via LaTeX.

    WARNING: Only call this tool after the user has explicitly approved the cover letter content!

    ``output_dir`` is the directory to write into (a session folder inside
    data/); it defaults to data/ itself.
    """
    try:
        if not output_filename.lower().endswith(".pdf"):
            return "Error: Output filename must end with .pdf"

        try:
            content = json.loads(content_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format for cover letter content. {str(e)}"

        required_fields = [
            "opening_paragraph",
            "body_paragraph_1",
            "body_paragraph_2",
            "closing_paragraph",
        ]
        missing_fields = [field for field in required_fields if field not in content]
        if missing_fields:
            return (
                "Error: Cover letter content is missing required fields: "
                f"{', '.join(missing_fields)}"
            )

        if not (recipient_info and recipient_info.strip()):
            return (
                "Error: recipient_info is required. Extract it from the job posting "
                "or ask the user who the letter should be addressed to."
            )

        cv_data: dict[str, Any] = {}
        job_data: dict[str, Any] | None = None
        if cv_json and cv_json.strip():
            try:
                cv_data = json.loads(cv_json)
            except json.JSONDecodeError:
                cv_data = {}
        if job_json and job_json.strip():
            try:
                job_data = json.loads(job_json)
            except json.JSONDecodeError:
                job_data = None

        tex_content = render_cover_letter_tex(
            content,
            applicant_name=applicant_name,
            applicant_contact=applicant_contact,
            recipient_info=recipient_info,
            cv_data=cv_data if isinstance(cv_data, dict) else {},
            job_data=job_data if isinstance(job_data, dict) else None,
        )
        target_dir = _resolve_output_dir(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(output_filename).stem
        pdf_path = compile_tex_to_pdf(tex_content, target_dir, stem)
        final_path = target_dir / output_filename
        if pdf_path != final_path:
            pdf_path.replace(final_path)

        return (
            f"Cover letter PDF generated successfully! "
            f"The file '{output_filename}' is ready for download."
        )

    except RuntimeError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error generating cover letter PDF: {str(e)}"


# ============================================
# Enhanced Tailoring Tools (Phase 1 & 2)
# ============================================

@tool
def generate_job_summary(job_json: str) -> str:
    """
    Generate a brief summary of the job posting including role, responsibilities, and required skills.
    
    Args:
        job_json: JobRequirements JSON string from job_agent output
        
    Returns:
        JSON string with job summary containing:
        - role: Job title and level
        - responsibilities: Key responsibilities
        - required_skills: List of required skills
        - preferred_skills: Nice-to-have skills
        - key_notes: Other important points
    """
    llm = chat_model("extract", temperature=0)
    structured_llm = llm.with_structured_output(JobSummary)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are summarizing a job posting. "
            "Extract the role title with seniority level, all key responsibilities, "
            "required skills, preferred/nice-to-have skills, and any other important "
            "notes (location, employment type, culture fit signals, etc.)."
        )),
        ("user", "Job posting data:\n{job}")
    ])

    try:
        chain = prompt | structured_llm
        summary: JobSummary = chain.invoke({"job": job_json})
        return summary.model_dump_json()
    except Exception as e:
        return json.dumps({
            "role": "Unknown",
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "key_notes": [f"Summarization failed: {str(e)[:120]}"]
        })

def _extract_compatibility_report(compatibility_data: Dict[str, Any]) -> Optional[CompatibilityReport]:
    """
    Accept either a v2 CompatibilityReport directly or the legacy shim shape
    (with `report_v2` embedded). Returns None if the input is malformed.
    """
    if not isinstance(compatibility_data, dict):
        return None
    candidate = compatibility_data
    if "report_v2" in compatibility_data and isinstance(compatibility_data["report_v2"], dict):
        candidate = compatibility_data["report_v2"]
    try:
        if "aggregate_score" in candidate or "dimensions" in candidate:
            return CompatibilityReport.model_validate(candidate)
    except Exception:
        return None
    return None


def _intensity_for_level(level: str) -> str:
    return {
        "excellent": "minor",
        "high": "minor",
        "medium": "moderate",
        "low": "major",
        "unknown": "moderate",
    }.get(level, "moderate")


def _build_directives_from_report(report: CompatibilityReport) -> Dict[str, Any]:
    """
    Translate a CompatibilityReport into concrete per-section directives the
    downstream tailoring tools can act on without needing further LLM reasoning.
    """
    gap = report.gap_analysis
    directives: List[Dict[str, Any]] = []
    focus_areas: List[str] = []

    # Hard skills: point the writer at the adjacent experience the CV already
    # has. No drafted bullet text goes in here - the scorer's bridge_bullet is
    # a claim about the required skill, and handing it to the rewriter as an
    # instruction is how a skill the CV never mentions ends up on it.
    adjacent = gap.transferable_skills + gap.learnable_skills
    for sk in adjacent:
        if not sk.matched_with:
            continue
        directives.append({
            "type": "surface_adjacent_experience",
            "section": "experience_or_projects",
            "required_skill": sk.required_skill,
            "candidate_skill": sk.matched_with,
            "guidance": (
                f"Where an existing bullet already shows {sk.matched_with}, make that experience "
                f"explicit. Do not claim {sk.required_skill}: the CV does not show it."
            ),
        })

    if gap.matched_skills:
        focus_areas.append("emphasize_matched_skills")
        directives.append({
            "type": "emphasize_skills_section",
            "skills_to_surface": [m.matched_with or m.required_skill for m in gap.matched_skills],
            "rationale": "These job-required skills are explicitly held by the candidate; surface them prominently.",
        })

    if any(sk.matched_with for sk in adjacent):
        focus_areas.append("surface_adjacent_experience")

    if gap.missing_skills:
        focus_areas.append("address_missing_skills")
        omit_count = sum(1 for m in gap.missing_skills if m.transferability < 0.2)
        soften_count = len(gap.missing_skills) - omit_count
        directives.append({
            "type": "handle_missing_skills",
            "missing": [m.required_skill for m in gap.missing_skills],
            "guidance": (
                f"Omit unfamiliar tooling that has no transferable foothold "
                f"({omit_count} skills); for the remaining {soften_count}, "
                "use neutral phrasing that emphasizes adjacent experience without overclaiming."
            ),
        })

    # Per-dimension directives
    dim_lookup = {d.name: d for d in report.dimensions}
    if dim_lookup.get("experience") and dim_lookup["experience"].score < 0.5:
        focus_areas.append("amplify_experience_outcomes")
        directives.append({
            "type": "amplify_outcomes",
            "section": "experience",
            "guidance": "Surface quantified outcomes (scope, scale, impact) in the most relevant roles to compensate for the years gap.",
        })
    if dim_lookup.get("seniority") and dim_lookup["seniority"].score < 0.5:
        focus_areas.append("strengthen_seniority_signals")
        directives.append({
            "type": "strengthen_seniority_language",
            "section": "experience",
            "guidance": "Where truthful, use scope/leadership verbs (led, owned, mentored, drove) and add team/scope sizing.",
        })
    # Exact posting wording only for skills the CV already has (e.g. the CV
    # says "Postgres", the posting says "PostgreSQL"); inserting required-skill
    # terms the CV lacks would be keyword stuffing of skills the candidate
    # doesn't hold.
    posting_terms = [m.required_skill for m in gap.matched_skills if m.required_skill]
    if dim_lookup.get("ats_keywords") and dim_lookup["ats_keywords"].score < 0.55 and posting_terms:
        focus_areas.append("use_posting_terms")
        directives.append({
            "type": "use_posting_terms",
            "section": "skills_and_summary",
            "skills": posting_terms,
            "guidance": (
                "Where the CV already shows these skills, use the posting's exact wording for them. "
                "Never add a skill the CV does not already show."
            ),
        })
    if dim_lookup.get("domain") and dim_lookup["domain"].score < 0.5:
        focus_areas.append("reframe_domain_language")
        directives.append({
            "type": "reframe_domain",
            "section": "summary_and_titles",
            "guidance": "Reframe role titles and summary to use the job's domain terminology where the candidate has matching experience.",
        })

    if gap.over_qualified_signals:
        focus_areas.append("temper_over_qualification")
        directives.append({
            "type": "temper_seniority",
            "guidance": "Keep senior-only details concise; emphasize hands-on contribution to avoid signaling over-qualification.",
            "signals": gap.over_qualified_signals,
        })

    return {
        "scoring_version": "v2",
        "aggregate_score": report.aggregate_score,
        "level": report.level,
        "intensity": _intensity_for_level(report.level),
        "strategy": report.interpretation,
        "focus_areas": list(dict.fromkeys(focus_areas)),  # de-duped, stable order
        "directives": directives,
        "summary": {
            "matched_skills_count": len(gap.matched_skills),
            "transferable_skills_count": len(gap.transferable_skills),
            "learnable_skills_count": len(gap.learnable_skills),
            "missing_skills_count": len(gap.missing_skills),
            "over_qualified_signals_count": len(gap.over_qualified_signals),
        },
    }


@tool
def decide_tailoring_strategy(compatibility_score_json: str, cv_json: str, job_json: str) -> str:
    """
    Decide tailoring strategy from a CompatibilityReport (v2) or the legacy
    compatibility score response.

    Produces concrete per-skill directives:
    - surface_adjacent_experience for each transferable skill (no drafted
      bullet text, and never a claim of the required skill itself)
    - emphasize_skills_section listing directly matched skills
    - handle_missing_skills with omit/soften guidance
    - amplify_outcomes / strengthen_seniority_language / reframe_domain when
      those dimensions score low
    - use_posting_terms (exact posting wording, matched skills only) when the
      ATS keyword dimension scores low
    - temper_seniority if over-qualification signals were detected

    These directives are consumed directly by select_prioritize_content and
    rewrite_enhance_content - no further LLM reasoning required to pick a
    strategy.

    Args:
        compatibility_score_json: JSON from calculate_compatibility_score (v1
            shim) or calculate_compatibility_score_v2.
        cv_json: ResumeInfo JSON string.
        job_json: JobRequirements JSON string.

    Returns:
        JSON string with strategy, intensity, focus_areas, and a structured
        directives list.
    """
    try:
        compatibility_data = json.loads(compatibility_score_json) if isinstance(compatibility_score_json, str) else compatibility_score_json
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid compatibility JSON: {str(e)}",
            "strategy": "standard",
            "intensity": "moderate",
            "focus_areas": [],
            "directives": [],
        })

    report = _extract_compatibility_report(compatibility_data)

    # If we couldn't parse a v2 report (e.g. caller passed a raw v1 dict from
    # a non-shim source), recompute v2 on the fly so we always have a rich
    # structure to act on.
    if report is None:
        try:
            cv_data = json.loads(cv_json) if isinstance(cv_json, str) else cv_json
            job_data = json.loads(job_json) if isinstance(job_json, str) else job_json
            if isinstance(cv_data, dict) and isinstance(job_data, dict):
                report = _calculate_compatibility_score_v2_internal(cv_data, job_data)
        except Exception:
            report = None

    if report is None:
        # Last-resort fallback to the old level-based defaults
        score = 0.0
        level = "unknown"
        if isinstance(compatibility_data, dict):
            score = float(compatibility_data.get("compatibility_score", 0.0) or 0.0)
            level = str(compatibility_data.get("level", "unknown"))
        fallback = {
            "scoring_version": "fallback",
            "aggregate_score": score,
            "level": level,
            "intensity": _intensity_for_level(level),
            "strategy": "Could not parse compatibility report; using level-based default.",
            "focus_areas": ["skills", "keywords", "section_ordering"],
            "directives": [],
        }
        return json.dumps(fallback)

    return json.dumps(_build_directives_from_report(report), indent=2)


def _format_directives(directives: List[Any]) -> str:
    """
    Render step 3's directives as one readable line each for a prompt.

    decide_tailoring_strategy emits dicts ({"type": ..., "guidance": ...}). A
    bare "\\n".join() on them raised TypeError before the LLM was ever called,
    and the tools' blanket except turned that into an empty result.
    """
    lines: List[str] = []
    for directive in directives or []:
        if isinstance(directive, dict):
            details = "; ".join(
                f"{key}: {', '.join(str(v) for v in value) if isinstance(value, list) else value}"
                for key, value in directive.items()
                if key != "type" and value not in (None, "", [])
            )
            kind = directive.get("type", "directive")
            lines.append(f"- {kind}: {details}" if details else f"- {kind}")
        elif directive:
            lines.append(f"- {directive}")
    return "\n".join(lines) if lines else "None provided."


def _format_plan(strategy_data: Dict[str, Any], step: str) -> str:
    """The parts of the user-approved strategist plan this step acts on.
    Keywords are only those validation found on the CV."""
    plan = strategy_data.get("plan")
    if not isinstance(plan, dict):
        return "None."
    lines = [f"Positioning: {plan.get('positioning', '')}"]
    if step == "select":
        if plan.get("section_order"):
            lines.append(f"Section order: {', '.join(plan['section_order'])} (use it as `section_order`)")
        if plan.get("bullets_to_feature"):
            lines.append(f"Feature these bullets: {', '.join(plan['bullets_to_feature'])}")
        if plan.get("project_order"):
            lines.append(f"Project order (indices, best first): {plan['project_order']}")
    else:
        if plan.get("headline"):
            lines.append(f"Headline idea for the summary's opening (the CV prints no separate headline): "
                         f"{plan['headline']}")
        keywords = [k for k in plan.get("keyword_plan") or [] if isinstance(k, dict)]
        usable = [k["keyword"] for k in keywords if k.get("allowed")]
        unusable = [k["keyword"] for k in keywords if not k.get("allowed")]
        if usable:
            lines.append(f"Posting terms the CV supports (use where they fit naturally): {', '.join(usable)}")
        if unusable:
            lines.append(f"Posting terms the CV does NOT support (never use): {', '.join(unusable)}")
    return "\n".join(lines)


def _format_feedback(strategy_data: Dict[str, Any], stages: tuple[str, ...]) -> str:
    """
    The user's (and the verifier's) notes for this step, oldest first.

    Gate edits and chat notes are appended to the strategy's ``user_feedback``
    as {stage, text}; without this they reached the step's state but never
    its prompt, so "edit with feedback" re-ran the step unchanged.
    """
    notes = [
        str(entry.get("text", "")).strip()
        for entry in strategy_data.get("user_feedback") or []
        if isinstance(entry, dict) and entry.get("stage") in stages and str(entry.get("text", "")).strip()
    ]
    return "\n".join(f"- {note}" for note in notes) if notes else "None."


@tool
def select_prioritize_content(
    cv_json: str,
    job_json: str,
    tailoring_strategy_json: str,
    bullet_index_block: str = "",
    evidence_block: str = "",
) -> str:
    """
    Select and prioritize the most relevant resume content based on job requirements.

    Args:
        cv_json: ResumeInfo JSON string
        job_json: JobRequirements JSON string
        tailoring_strategy_json: JSON string from decide_tailoring_strategy
        bullet_index_block: Numbered "[bullet_id] text" list of every bullet
            in the CV. When provided, each selected bullet must echo the
            matching bullet_id verbatim so downstream steps can address it.
        evidence_block: The requirement rubric as "[req.N] category skill
            (status): bullet ids that show it" lines, from the scorer.

    Returns:
        JSON string with selected bullets, recommended section order, and emphasis plan
        (fields: selected_bullets[{section, original_text, relevance_score, reason, bullet_id}],
         section_order, sections_to_emphasize, items_to_de_emphasize)
    """
    try:
        strategy_data = json.loads(tailoring_strategy_json)
        focus_areas: List[str] = strategy_data.get("focus_areas", [])
        directives: List[Dict[str, Any]] = strategy_data.get("directives", [])

        llm = chat_model("writer", temperature=0)
        structured_llm = llm.with_structured_output(SelectedContent)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a career consultant selecting and prioritising resume content for a specific job. "
                "Your decisions must be driven by the tailoring strategy directives provided. "
                "CRITICAL RULES:\n"
                "- Only work with content that EXISTS in the CV — do not invent bullets.\n"
                "- Score each bullet 0–1 for relevance to the job requirements.\n"
                "- Recommend a section ordering that puts the most impactful sections first.\n"
                "- Mark low-value items for de-emphasis (do not delete — the candidate reviews first).\n"
                "- A numbered bullet index (`[bullet_id] text`) may be provided below. When it is, "
                "set `bullet_id` on every selected bullet to the exact id of the matching line — "
                "never invent an id or alter its text.\n"
                "- An evidence map may be provided: for each job requirement, the bullets that "
                "already show it. Prefer bullets that evidence must-have requirements, then "
                "nice-to-haves; when several bullets show the same requirement, the strongest one "
                "is usually enough. Say in `reason` which requirement a bullet evidences.\n"
                "- NEVER return an empty `selected_bullets` list if the CV has any bullets at all. "
                "Even on a weak or poor fit, select the CV's best-available bullets (aim for at "
                "least 3-5, or every bullet if the CV has fewer): the strongest, most quantified, "
                "most senior-sounding ones, and ones that use ANY vocabulary adjacent to the job's "
                "required/preferred skills, even loosely. Low relevance is still worth reporting as "
                "a low `relevance_score` with an honest `reason` (e.g. 'no direct overlap, included "
                "as the strongest available bullet') - it is never a reason to select nothing. The "
                "user relies on this step to see their CV's best foot forward, not silence."
            )),
            ("user", (
                "CV Data:\n{cv}\n\n"
                "Job Data:\n{job}\n\n"
                "Bullet Index:\n{bullet_index}\n\n"
                "Evidence Map (requirement -> bullets that show it):\n{evidence}\n\n"
                "Tailoring Plan (approved by the user):\n{plan}\n\n"
                "Focus Areas: {focus}\n\n"
                "Strategy Directives:\n{directives}\n\n"
                "User Feedback on earlier selections (follow it; it overrides the directives "
                "where they conflict):\n{feedback}\n\n"
                "Select and prioritise the most relevant content."
            ))
        ])

        chain = prompt | structured_llm
        result: SelectedContent = chain.invoke({
            "cv": cv_json,
            "job": job_json,
            "bullet_index": bullet_index_block or "(not provided)",
            "evidence": evidence_block or "(not provided)",
            "plan": _format_plan(strategy_data, "select"),
            "focus": ", ".join(focus_areas),
            "directives": _format_directives(directives),
            "feedback": _format_feedback(strategy_data, ("approve_selection", "chat")),
        })
        return result.model_dump_json()

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "selected_bullets": [],
            "section_order": [],
            "sections_to_emphasize": [],
            "items_to_de_emphasize": []
        })

@tool
def rewrite_enhance_content(
    cv_json: str,
    job_json: str,
    selected_content_json: str,
    tailoring_strategy_json: str,
    locked_bullets_json: str = "[]",
    facts_block: str = "",
) -> str:
    """
    Rewrite and enhance resume content to emphasize job relevance.

    Args:
        cv_json: Original ResumeInfo JSON string
        job_json: JobRequirements JSON string
        selected_content_json: JSON string from select_prioritize_content
        tailoring_strategy_json: JSON string from decide_tailoring_strategy
        locked_bullets_json: JSON list of {bullet_id, original, rewritten,
            confidence, keywords_added} for bullets the user has already
            accepted in a previous pass. These must be echoed back verbatim,
            unrewritten - only the remaining selected bullets are rewritten.
        facts_block: "[fact_id] value" lines from the fact ledger that a
            rewrite may cite in ``cited_fact_ids``.

    Returns:
        JSON string with rewritten bullets, updated summary, and per-bullet confidence scores
        (fields: rewritten_bullets[{original, rewritten, confidence, keywords_added, bullet_id}],
         updated_summary, keywords_inserted)
    """
    try:
        strategy_data = json.loads(tailoring_strategy_json)
        directives: List[Dict[str, Any]] = strategy_data.get("directives", [])
        try:
            locked_bullets = json.loads(locked_bullets_json) or []
        except json.JSONDecodeError:
            locked_bullets = []
        locked_ids = {b.get("bullet_id") for b in locked_bullets if isinstance(b, dict) and b.get("bullet_id")}

        llm = chat_model("writer", temperature=0.3)
        structured_llm = llm.with_structured_output(RewrittenContent)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a senior career coach rewriting resume bullets to maximise job relevance. "
                "Apply the strategy directives precisely.\n\n"
                "ABSOLUTE RULES:\n"
                "- Reword EXISTING content only — never fabricate experience, metrics, or skills.\n"
                "- Never add a skill, tool, or technology that the original bullet and CV do not "
                "already mention, even when a directive or the job asks for it.\n"
                "- Weave in job keywords naturally — forced keyword stuffing degrades ATS scores.\n"
                "- Lead each bullet with a strong action verb.\n"
                "- Write every rewrite and the summary in the language the CV is written in. The CV "
                "was already put into the language the candidate chose; never switch languages.\n"
                "- Write naturally, the way a native professional writes a CV in that language: "
                "idiomatic phrasing, no word-for-word constructions, no anglicisms in German.\n"
                "- The summary (`updated_summary`) describes the candidate's profile and strengths. "
                "Never state that they are looking for a job, a position or an internship ('seeking "
                "a working student role', 'sucht eine Werkstudentenstelle'): the application itself "
                "says that. If the CV has a summary, always return an improved `updated_summary`, "
                "with any such sentence removed.\n"
                "- Keep the candidate's authentic voice.\n"
                "- Only enhance descriptions — do not replace their substance.\n"
                "- For each bullet, report a confidence score (0–1) for how well the rewrite "
                "matches the job, and list keywords actually added.\n"
                "- Each selected bullet carries a `bullet_id`; copy it verbatim into the matching "
                "rewrite's `bullet_id` field.\n"
                "- Every technology, number and date in a rewrite must come from the original "
                "bullet or from a fact in the Facts list. When a rewrite uses something from the "
                "Facts list (e.g. a skill from the skills section), put that fact's id in "
                "`cited_fact_ids`; leave it empty when you only reword the original. Only cite a "
                "fact when it is plausibly true for that bullet's work. Rewrites are fact-checked "
                "against this list and anything unsupported is rejected.\n"
                "- Locked bullets (listed below, already approved by the user in an earlier pass) "
                "must be echoed back UNCHANGED — do not rewrite them again, just include them as-is "
                "in the output so the final list stays complete.\n"
                "- Produce one entry in `rewritten_bullets` for EVERY bullet in Selected Content "
                "(locked ones included). NEVER return an empty list when Selected Content is "
                "non-empty. If a bullet has no obvious job keyword to weave in, still tighten its "
                "wording and lead with a strong verb, and leave `keywords_added` empty for it — "
                "that is a valid, honest result. Doing nothing to a selected bullet is not."
            )),
            ("user", (
                "Original CV:\n{cv}\n\n"
                "Job Requirements:\n{job}\n\n"
                "Selected Content:\n{selected}\n\n"
                "Locked Bullets (echo back unchanged):\n{locked}\n\n"
                "Facts (what the CV and the candidate state; cite by id):\n{facts}\n\n"
                "Tailoring Plan (approved by the user):\n{plan}\n\n"
                "Strategy Directives:\n{directives}\n\n"
                "User Feedback on earlier rewrites (follow it; it overrides the directives "
                "where they conflict, but never the absolute rules):\n{feedback}\n\n"
                "Rewrite and enhance the content that is not locked."
            ))
        ])

        chain = prompt | structured_llm
        result: RewrittenContent = chain.invoke({
            "cv": cv_json,
            "job": job_json,
            "selected": selected_content_json,
            "locked": json.dumps(locked_bullets) if locked_bullets else "(none)",
            "facts": facts_block or "(not provided)",
            "plan": _format_plan(strategy_data, "rewrite"),
            "directives": _format_directives(directives),
            "feedback": _format_feedback(strategy_data, ("approve_rewrite", "chat", "verifier")),
        })

        # Belt-and-braces: force locked bullets back into the result verbatim
        # even if the model's echo drifted, then de-dupe by bullet_id/text.
        by_key: dict[str, Any] = {}
        for b in result.rewritten_bullets:
            key = b.bullet_id or b.original
            by_key[key] = b
        for locked in locked_bullets:
            if not isinstance(locked, dict):
                continue
            key = locked.get("bullet_id") or locked.get("original")
            if key:
                by_key[key] = BulletRewrite(**{
                    "original": locked.get("original", ""),
                    "rewritten": locked.get("rewritten", locked.get("original", "")),
                    "confidence": float(locked.get("confidence", 1.0)),
                    "keywords_added": locked.get("keywords_added", []) or [],
                    "bullet_id": locked.get("bullet_id"),
                    "cited_fact_ids": locked.get("cited_fact_ids", []) or [],
                })
        result.rewritten_bullets = list(by_key.values())

        return result.model_dump_json()

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "rewritten_bullets": [],
            "updated_summary": "",
            "keywords_inserted": []
        })

# ============================================
# Agent Configuration (removed)
# ============================================
#
# The legacy ``llm``, ``system_prompt``, ``agent`` (a ``create_agent`` instance)
# and ``run_interactive_writer`` CLI used to live below this line. They drove
# the entire 6-step tailoring workflow from a single giant system prompt.
#
# That sequencing is now expressed explicitly as graph nodes in
# ``app.agents.orchestrator.nodes_job_tailoring`` and the per-step prompts
# for the CV-review flow live in ``nodes_cv_review``. The orchestrator calls
# the ``@tool`` functions above directly; nothing in the runtime imports
# ``writer_agent.agent`` or ``writer_agent.system_prompt`` any more.
_AGENT_CONFIGURATION_REMOVED = True

_REMOVED_LEGACY_SYSTEM_PROMPT_HEAD = (
    "Removed. The legacy WriterAgent system prompt has been replaced by the "
    "per-step prompts in app.agents.orchestrator. See the module docstring."
)

_legacy_system_prompt_body = """legacy_removed
{
You are a professional CV and cover letter writer specializing in tailoring application materials to specific job opportunities.

IMPORTANT CONTEXT:
- You receive PRE-PROCESSED, STRUCTURED DATA from upstream agents:
  * CV data (ResumeInfo JSON) - extracted by cv_agent
  * Job requirements (JobRequirements JSON) - extracted by job_agent  
  * Company info (CompanyInfo JSON, optional) - researched by job_agent
- Your role is to TRANSFORM this data into tailored, professional documents
- You DO NOT extract data from raw files - that's already done
- The complete CV and job data are provided in a system message at the start of our conversation
- This data persists throughout our entire conversation - always use the FULL JSON data from the system context when calling tools
- Do NOT use truncated or partial data from earlier messages - always reference the complete data from the system context

YOUR WORKFLOW (FOLLOW THIS SEQUENCE):

ENHANCED TAILORING MODE (when both CV and job data are provided):
Use this comprehensive 6-step workflow for optimal results:

STEP 1: INPUT & SUMMARIES
   - Display the extracted CV information to the user (show what was extracted, no summary needed)
   - Call generate_job_summary to create a brief summary of the job posting
   - Present both: the CV data overview and the job summary (role, responsibilities, required skills)
   - FORMATTING: You may use ## or ### headers, NO code blocks, NO lists with dashes (-)
   - Use plain text with line breaks for lists
   - Example format:
     "Overview of Resume and Job Match
     
     Job Title: [title]
     
     Responsibilities:
     [responsibility 1]
     [responsibility 2]
     
     Required Skills:
     Technical Skills: [list as plain text, separated by commas or line breaks]
     Soft Skills: [list as plain text, separated by commas or line breaks]"
   - This helps verify extraction accuracy and sets context

STEP 2: COMPATIBILITY ANALYSIS
   - Call calculate_compatibility_score with CV and job data
   - This runs the v2 multi-dimensional scorer which measures five weighted dimensions:
     * Hard Skills match (40%): per-skill transferability via direct/family/embedding/LLM cascade
     * Experience alignment (15%): years and depth of relevant experience
     * Seniority fit (15%): level match between candidate and role
     * Domain overlap (15%): industry/domain semantic similarity
     * ATS keyword density (15%): exact-keyword coverage for ATS filters
   - The tool also returns a gap analysis bucketed into: matched skills, transferable skills (with bridge bullets), and missing skills
   - Present the aggregate compatibility score, level, per-dimension breakdown, and key gaps
   - FORMATTING: You may use ## or ### headers, NO code blocks, NO lists with dashes
   - Use plain text with line breaks, colon format for labels
   - Example format:
     "Compatibility Analysis

     Compatibility Score: 0.72 (HIGH)

     Dimension Breakdown:
     Hard Skills (40%): 0.80 — strong direct and family matches
     Experience (15%): 0.70 — relevant depth present
     Seniority (15%): 0.75 — level aligns well
     Domain (15%): 0.65 — adjacent domain, some reframing needed
     ATS Keywords (15%): 0.60 — moderate keyword density

     Key Gaps:
     Transferable: Vue → React (bridge: 'Built component-driven UIs with Vue; transferable to React')
     Missing: Kubernetes (no equivalent found)

     Interpretation: Strong match. Focus on adding bridge bullets for transferable skills and injecting missing keywords."
   - Interpretation guidelines:
     * LOW (< 0.5): Many required skills/experience missing — flag before proceeding
     * MEDIUM (0.5 - 0.7): Good match with clear areas to improve
     * HIGH (> 0.7): Strong match, fine-tune and optimise
     * EXCELLENT (> 0.85): Near-perfect match, polish only
   - ERROR HANDLING: If score is 0.0 or error occurs:
     * Check if CV or job data is missing/empty
     * Try calling the tool again with valid data
     * If still failing, explain: "The compatibility calculation encountered an issue. This may be due to missing or incomplete data. Let's proceed with manual analysis instead."
     * Then use analyze_cv_job_alignment as fallback
   - AFTER presenting the score and dimension breakdown, always add a plain-language "Weak Points" section.
     This section MUST appear before Step 3 and MUST be written in plain, jargon-free sentences — no scores,
     no percentages, no technical labels. Identify the 2–4 most significant gaps and state them directly.
     Rules:
     * For each missing skill: one sentence naming the skill and why it matters for this role.
     * For each low-scoring dimension (< 0.6): one sentence explaining what the gap means in practice.
     * For transferable skills: one sentence noting the gap and the bridge that exists.
     * Do NOT pad this section — if there are no meaningful weak points, say so in one sentence.
   - Example Weak Points format:
     "Weak Points

     You have no listed experience with Kubernetes, which is a hard requirement for container
     orchestration in this role — this is the most significant gap.

     Your background is primarily in e-commerce; the role targets fintech, so some reframing of
     domain experience will be needed.

     You have not worked at the senior/lead level before; the role expects team leadership, which
     is not evidenced in your CV.

     Your Vue.js experience is transferable to React, but you will need to surface that connection
     explicitly — a bridge bullet in your experience section will help."

STEP 3: DECIDE TAILORING STRATEGY
   - Call decide_tailoring_strategy with compatibility score, CV, and job data
   - This determines:
     * Strategy: What approach to take (highlight transferable skills, reorder sections, fine-tune)
     * Intensity: How much editing needed (major/moderate/minor)
     * Focus areas: Which sections to prioritize (skills, projects, section_ordering, keywords, etc.)
   - BEFORE presenting the strategy details, open with a brief bridging paragraph that references
     the weak points identified in Step 2 and states explicitly how the strategy will address each one.
     Keep this to 2–4 sentences. Example:
     "Based on the weak points above, here is how the tailoring strategy will address them:
     The missing Kubernetes experience will be handled by injecting the keyword in your skills section
     and noting adjacent container experience in your project bullets. The fintech domain gap will be
     addressed by reframing your payments and fraud-detection work. The leadership gap cannot be
     fabricated — we will surface any mentoring or cross-functional coordination already in your CV."
   - Then present the full strategy
   - FORMATTING: You may use ## or ### headers, NO code blocks
   - Example format:
     "Tailoring Strategy

     Strategy: [strategy description]
     Intensity: [major/moderate/minor]
     Focus Areas: [list as plain text, separated by commas]"
   - If compatibility is LOW, inform user and ask if they want to proceed (may need to highlight transferable skills)

STEP 4: SELECT & PRIORITIZE CONTENT
   - Call select_prioritize_content with CV, job, and tailoring strategy
   - This identifies:
     * Top N bullets per section based on relevance
     * New ordering for sections and items
     * Which sections to emphasize
     * Items to de-emphasize or remove (if any)
   - Present the selected content and proposed ordering
   - FORMATTING: You may use ## or ### headers, NO code blocks, NO lists with dashes
   - Use plain text with line breaks, numbered items as "1. " format
   - WAIT for user approval before proceeding

STEP 5: REWRITE & ENHANCE CONTENT
   - Call rewrite_enhance_content with CV, job, selected content, and strategy
   - This creates:
     * Rewritten bullets with original and new versions
     * Enhanced summary/headline (if applicable)
     * Confidence scores per bullet (0-1) showing job match quality
     * List of job keywords naturally incorporated
   - Present the rewritten content with confidence scores
   - FORMATTING: You may use ## or ### headers, NO code blocks
   - Show what changed and why in clear, readable format
   - Example format:
     "Rewritten Content
     
     Original: [bullet text]
     Enhanced: [rewritten bullet text]
     Confidence: 0.85 (high match)
     
     Changes: [explanation of what changed and why]"
   - WAIT for user approval

STEP 6: USER REVIEW & EXPORT
   - Merge the rewritten content from Step 5 back into the CV JSON structure:
     * Update experience bullets with rewritten versions
     * Update summary/headline if enhanced
     * Apply new ordering from Step 4
   - Call generate_tailored_cv_html with the updated CV JSON and a tailoring plan (can create a simple plan from the strategy)
   - Present the complete tailored resume HTML for review
   - Show what changed from the original
   - Ask user if they're satisfied
   - If approved, proceed to document generation:
     * Ask user: "Ready to generate the document? I can create a PDF, Word document (.docx), or both formats. Which would you prefer?"
     * Call generate_cv_pdf and/or generate_cv_docx with the HTML content
   - If changes needed, incorporate feedback and repeat Step 5

STANDARD WORKFLOW (alternative, simpler approach):

1. ANALYSIS PHASE:
   - When user provides CV and job data, call analyze_cv_job_alignment
   - This generates a tailoring plan showing how to optimize the CV

2. REVIEW PHASE:
   - PRESENT the tailoring plan to the user in clear, readable format
   - IMPORTANT: Avoid complex markdown structure (no ### or deeper headers, no - lists, no code blocks)
   - You may use simple formatting: ## for section headers, **bold** for emphasis, and *italic* for subtle emphasis
   - Use simple line breaks, indentation, and numbered/bullet points written as plain text
   - Explain the key recommendations
   - WAIT for user feedback before proceeding
   - User might say:
     * "looks good" or "approved" → proceed to generation
     * "emphasize X more" → note the adjustment
     * "don't mention Y" → note the constraint
     * "redo the analysis" → call the tool again with updated context
   - After refining or discussing each section (Education, Experience, Skills, Projects, etc.), ask if they want to generate the document for that section
   - You can refine multiple sections before generating, but always offer generation after showing each refined section

3. CV GENERATION PHASE:
   - After approval, call generate_tailored_cv_html
   - SHOW a preview/summary of the generated HTML to the user
   - After showing each section or the complete CV, ALWAYS ask: "Ready to generate the document? I can create a PDF, Word document (.docx), or both formats. Which would you prefer?"
   - If user requests "both", "pdf and word", "both formats", or similar, generate BOTH formats
   - If user requests "pdf" or "PDF", generate only PDF
   - If user requests "word", "docx", or "Word", generate only Word document
   - If user doesn't specify, ask for clarification
   - ONLY call generate_cv_pdf and/or generate_cv_docx after EXPLICIT approval
   - Use appropriate filename (e.g., "firstname_lastname_cv_tailored.pdf" or ".docx")
   - When generating both formats, call BOTH tools and include BOTH success messages in your response
   - Note: Word documents have minimal formatting so users can easily customize them

4. COVER LETTER PHASE (if requested):
   - Before calling generate_cover_letter_content, ask the user which language/format they want:
     "Would you like the cover letter in English (standard cover letter) or German (formal Anschreiben in Sie-form)?"
     * If the user says "English" or doesn't specify: pass language="english"
     * If the user says "German", "Deutsch", "Anschreiben", or similar: pass language="german"
   - Call generate_cover_letter_content with CV, job, company data, and the chosen language parameter
   - SHOW the cover letter content to the user
     * For German: show Betreff, Anrede, and all paragraphs including Grußformel
     * For English: show opening/body/closing paragraphs as before
   - Ask for feedback or approval
   - After showing the cover letter, ALWAYS ask: "Ready to generate the document? I can create a PDF, Word document (.docx), or both formats. Which would you prefer?"
   - If user requests "both", "pdf and word", "both formats", or similar, generate BOTH formats
   - If user requests "pdf" or "PDF", generate only PDF
   - If user requests "word", "docx", or "Word", generate only Word document
   - If user doesn't specify, ask for clarification
   - ONLY call generate_cover_letter_pdf and/or generate_cover_letter_docx after EXPLICIT approval
   - When calling generate_cover_letter_pdf or generate_cover_letter_docx, you MUST provide:
     * content_json: The output from generate_cover_letter_content
     * output_filename: e.g., "firstname_lastname_cover_letter.pdf" or "firstname_lastname_anschreiben.pdf" for German
     * applicant_name: Extract from CV data (e.g., cv_data['name'])
     * applicant_contact: Format as "email | phone" from CV data (e.g., "john@email.com | (123) 456-7890")
     * recipient_info: Extract recipient_name from job data when present; otherwise ask the user. Never use a generic default.
   - When generating both formats, call BOTH tools and include BOTH success messages in your response

CRITICAL PRINCIPLES:
ALWAYS show changes before generating PDFs
WAIT for explicit user approval at each checkpoint
Preserve the candidate's authentic voice and style
Only emphasize/refine EXISTING content - never fabricate
Be transparent about what you're changing and why
Respect user constraints and preferences
If uncertain, ask clarifying questions

NEVER generate PDFs without approval
NEVER fabricate experience, skills, or achievements
NEVER proceed to next phase without user confirmation
NEVER ignore user feedback or constraints

WHEN TO USE ENHANCED VS STANDARD WORKFLOW:
- Use ENHANCED TAILORING MODE when both CV and job data are provided and user wants comprehensive analysis
- Use STANDARD WORKFLOW for simpler, faster tailoring or when user prefers a quicker process
- You can offer both options: "I can use the enhanced tailoring mode for detailed analysis, or the standard workflow for faster results. Which would you prefer?"
- Enhanced mode provides: compatibility scores, semantic analysis, detailed strategy, confidence scores
- Standard mode provides: quick gap analysis and tailoring plan

ENHANCED TOOLS REFERENCE:
- generate_job_summary: Creates structured job summary (role, responsibilities, required/preferred skills) — Step 1
- calculate_compatibility_score: Multi-dimensional v2 scorer — 5 weighted dimensions (hard_skills 40%, experience/seniority/domain/ats_keywords 15% each) plus skill-level gap analysis — Step 2
  NOTE: calculate_semantic_similarity and calculate_bm25_score are sub-components used internally; call calculate_compatibility_score directly for Step 2 to get the full report.
- decide_tailoring_strategy: Consumes the v2 CompatibilityReport and emits per-skill directives (inject_bridge_bullet, emphasize_skills_section, handle_missing_skills, amplify_outcomes, inject_keywords, reframe_domain, temper_seniority) — Step 3
- select_prioritize_content: Selects top bullets per section with relevance scores, recommends section ordering and emphasis — Step 4
- rewrite_enhance_content: Rewrites selected bullets with action verbs and job keywords; returns per-bullet confidence scores — Step 5
- generate_cover_letter_content: Generates structured cover letter content. Accepts an optional language parameter:
  language="english" (default) → standard English cover letter (opening/body/closing paragraphs)
  language="german" → formal German Anschreiben in Sie-form (betreff, Einleitung, Hauptteil, Schlussteil, Grußformel)
  Always ask the user which language they want before calling this tool.

COMPATIBILITY THRESHOLDS:
- LOW: < 0.5 - Many gaps, consider transferable skills
- MEDIUM: 0.5 - 0.7 - Good match, optimize ordering and keywords
- HIGH: > 0.7 - Strong match, fine-tune and optimize

INTERACTION STYLE:
- Professional yet friendly
- Clear and concise in explanations
- Proactive in asking for clarification
- Transparent about your process
- Collaborative, not autonomous

CRITICAL FORMATTING RULES:
- You MAY use ## headers (two hashes) for major sections
- You MAY use ### headers (three hashes) for subsections
- NEVER use code blocks (triple backticks)
- NEVER use markdown lists with dashes (-) - use plain text with line breaks instead
- You MAY use **bold** for emphasis and *italic* for subtle emphasis
- Use plain text with line breaks for lists
- Use numbered format "1. " for numbered lists
- Format responses with simple line breaks and plain text structure
- When presenting compatibility scores, use clear format: "Compatibility Score: 0.72 (HIGH)"
- When showing data, use simple colon format: "Job Title: [title]" not markdown tables

**CRITICAL - File Downloads:**
- When you call generate_cv_pdf, generate_cv_docx, generate_cover_letter_pdf, or generate_cover_letter_docx tools, they return a success message
- You MUST include this EXACT success message in your response to the user
- DO NOT paraphrase or reword the tool's success message - copy it verbatim
- The success message contains the filename which is needed for the download system to work
- DO NOT create markdown links like [Download](sandbox:/filename.pdf) or [Download](file://filename.pdf)
- DO NOT create any file://, sandbox:/, or other file links - they don't work
- The user interface will AUTOMATICALLY display download buttons when it detects the tool's success messages
- If you generate BOTH formats (PDF and Word), include BOTH success messages and the UI will show TWO download buttons
- You can add additional text before or after the tool messages, but each tool's exact message MUST be included
- For Word documents, mention that formatting is minimal so users can customize it easily

Examples:
- Single format: Tool returns "CV PDF generated successfully! The file 'john_doe_cv.pdf' is ready for download."
  Your response: "Great! CV PDF generated successfully! The file 'john_doe_cv.pdf' is ready for download."
- Both formats: 
  Tool 1 returns: "CV PDF generated successfully! The file 'john_doe_cv.pdf' is ready for download."
  Tool 2 returns: "CV Word document generated successfully! The file 'john_doe_cv.docx' is ready for download."
  Your response MUST include BOTH messages: "Perfect! I've generated both formats for you. CV PDF generated successfully! The file 'john_doe_cv.pdf' is ready for download. CV Word document generated successfully! The file 'john_doe_cv.docx' is ready for download. You can customize the Word document formatting in Microsoft Word or Google Docs."

Remember: legacy system prompt removed; orchestrator drives sequencing now.
"""


# The legacy ``create_agent`` instance, top-level ``llm`` / ``system_prompt``
# bindings, ``run_interactive_writer`` CLI runner, and ``__main__`` entry point
# have all been removed. The orchestrator graph nodes call the @tool
# functions defined above directly, and per-step prompts now live next to
# their respective nodes in ``app.agents.orchestrator``.
_LEGACY_AGENT_REMOVED = True


def _legacy_run_interactive_writer_removed(*_args, **_kwargs):
    """Stub. The CLI runner has been removed; use the orchestrator API.

    The remainder of this module file has been intentionally truncated. The
    real terminator below ensures everything after the stub is dead code.
    """
    raise RuntimeError(
        "run_interactive_writer has been removed. "
        "Use the orchestrator graph (app.agents.orchestrator.build_graph) or "
        "the REST API at /api/orchestrator/* instead."
    )


# ---------------------------------------------------------------------------
# Everything below is the old legacy CLI body, wrapped in a never-called
# function so undefined names (agent, etc.) are inert. We keep it instead of
# re-emitting the entire 1900-line file just to delete the tail.
# ---------------------------------------------------------------------------

def _legacy_interactive_writer_body_REMOVED(  # pragma: no cover
    cv_json: str = None,
    job_json: str = None,
    company_json: str = None,
):
    _legacy_tail_docstring = """
    Run writer agent interactively in terminal for testing.
    
    This function provides a CLI interface to test the writer agent with
    mock or real data from cv_agent and job_agent outputs.
    
    Args:
        cv_json: Optional ResumeInfo JSON string (if None, prompts user)
        job_json: Optional JobRequirements JSON string (if None, prompts user)
        company_json: Optional CompanyInfo JSON string (can be None)
        
    Returns:
        Dict with conversation history and results
        
    Example:
        >>> # With data
        >>> run_interactive_writer(cv_data, job_data)
        
        >>> # Interactive input
        >>> run_interactive_writer()
    """
    print("=" * 70)
    print("WRITER AGENT - Interactive Testing Mode")
    print("=" * 70)
    print("\nThis agent helps you tailor CVs and cover letters to job opportunities.")
    print("It works with structured JSON data from cv_agent and job_agent.\n")
    
    # Get data if not provided
    if not cv_json:
        print("Enter path to CV JSON file (or 'mock' for test data):")
        cv_input = input("> ").strip()
        if cv_input.lower() == 'mock':
            cv_json = json.dumps({
                "name": "Test Candidate",
                "email": "test@example.com",
                "phone": "+1234567890",
                "skills": ["Python", "JavaScript", "React"],
                "education": ["B.Sc. Computer Science"],
                "experience": ["Software Developer at Tech Corp"]
            })
        else:
            with open(cv_input, 'r') as f:
                cv_json = f.read()
    
    if not job_json:
        print("\nEnter path to Job JSON file (or 'mock' for test data):")
        job_input = input("> ").strip()
        if job_input.lower() == 'mock':
            job_json = json.dumps({
                "job_title": "Senior Software Engineer",
                "job_level": "Senior",
                "required_skills": ["Python", "React", "AWS"],
                "responsibilities": ["Build scalable applications"],
                "qualifications": ["Bachelor's degree in CS"]
            })
        else:
            with open(job_input, 'r') as f:
                job_json = f.read()
    
    # Initialize conversation
    initial_message = f"""I have CV and job data ready for tailoring.

CV Data:
{cv_json[:200]}...

Job Data:
{job_json[:200]}...

Please analyze the alignment and help me create a tailored CV."""
    
    messages = [{"role": "user", "content": initial_message}]
    
    # Conversation loop
    max_turns = 20
    turn = 0
    
    print("\n" + "=" * 70)
    print("Starting Conversation (type 'quit' to exit)")
    print("=" * 70)
    
    while turn < max_turns:
        turn += 1
        print(f"\n{'='*70}")
        print(f"Turn {turn}")
        print(f"{'='*70}\n")
        
        # Agent's turn
        print("🤖 Agent is processing...\n")
        try:
            result = agent.invoke({"messages": messages})
            agent_response = result["messages"][-1].content
            
            # Add to history
            messages.append({"role": "assistant", "content": agent_response})
            
            # Display response
            print(f"Agent:\n{agent_response}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # User's turn
        print(f"{'-'*70}")
        user_input = input("You: ").strip()
        
        # Check for exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting writer agent.")
            break
        
        # Add user response
        messages.append({"role": "user", "content": user_input})
    
    if turn >= max_turns:
        print(f"\n⚠️  Reached maximum turns ({max_turns}).")
    
    return {
        "messages": messages,
        "turns": turn
    }

# The original ``if __name__ == "__main__":`` CLI entry point invoked the
# now-removed run_interactive_writer(). It has been deleted - use the
# orchestrator graph (app.agents.orchestrator) or the REST API to drive the
# writer tools instead.
