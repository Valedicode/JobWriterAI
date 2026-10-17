"""
Writer Agent API Endpoints

Stateless deterministic endpoints around the writer agent's ``@tool``
functions. The conversational chat surface has moved to
``/api/orchestrator/*`` (see ``app.api.orchestrator``); only the tool wrappers
needed by external callers and the deterministic CV/cover-letter generators
remain here.

The legacy ``RESUME_REFINEMENT_SYSTEM_PROMPT`` and Writer-agent ``create_agent``
loop used to drive the conversation directly from this router. Both have been
removed: per-step prompts now live next to their nodes in
``app.agents.orchestrator.nodes_cv_review`` and ``nodes_job_tailoring``.
"""

import json
import re
import uuid

from fastapi import APIRouter, HTTPException, status

from app.agents.writer_agent import (
    analyze_cv_job_alignment,
    generate_cover_letter_content,
    generate_cover_letter_pdf,
    generate_cv_pdf,
)
from app.services.file_store import UNSAFE_FILENAME_MESSAGE, get_file_store, is_safe_filename
from app.services.latex_renderer import extract_recipient_name
from app.models.schemas import (
    CoverLetterContent,
    CVJobAlignmentRequest,
    CVJobAlignmentResponse,
    CVTailoringPlan,
    GenerateCoverLetterRequest,
    GenerateCoverLetterResponse,
    GenerateTailoredCVRequest,
    GenerateTailoredCVResponse,
    JobRequirements,
    ResumeInfo,
)

router = APIRouter(prefix="/api/writer", tags=["Writer Agent"])


# ============================================
# /analyze-alignment
# ============================================

@router.post(
    "/analyze-alignment",
    response_model=CVJobAlignmentResponse,
    summary="Analyze CV-Job Alignment",
    description="Analyze how well a CV matches job requirements and create a tailoring plan."
)
def analyze_alignment(request: CVJobAlignmentRequest):
    """
    Analyze CV-job alignment and create a tailoring plan.

    Used by clients that want the tailoring plan as a standalone artifact
    rather than going through the full orchestrator graph.
    """
    try:
        try:
            ResumeInfo(**request.cv_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CV data structure: {str(e)}"
            )

        try:
            JobRequirements(**request.job_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job data structure: {str(e)}"
            )

        cv_json = json.dumps(request.cv_data)
        job_json = json.dumps(request.job_data)

        tailoring_plan_json = analyze_cv_job_alignment.invoke({
            "cv_json": cv_json,
            "job_json": job_json,
        })

        tailoring_plan_dict = json.loads(tailoring_plan_json)

        return CVJobAlignmentResponse(
            success=True,
            tailoring_plan=CVTailoringPlan(**tailoring_plan_dict),
            message="Alignment analysis completed successfully."
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse tailoring plan: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing alignment: {str(e)}"
        )


# ============================================
# /generate-cv
# ============================================

@router.post(
    "/generate-cv",
    response_model=GenerateTailoredCVResponse,
    summary="Generate Tailored CV",
    description="Generate a tailored CV PDF from a CV + tailoring plan pair."
)
def generate_cv(request: GenerateTailoredCVRequest):
    """Deterministic CV PDF generation. The orchestrator uses the underlying
    @tool functions directly; this REST endpoint stays for external callers."""
    try:
        try:
            ResumeInfo(**request.cv_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CV data: {str(e)}"
            )

        try:
            CVTailoringPlan(**request.tailoring_plan)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tailoring plan: {str(e)}"
            )

        if not is_safe_filename(request.output_filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=UNSAFE_FILENAME_MESSAGE,
            )

        cv_json = json.dumps(request.cv_data)
        section_order = request.tailoring_plan.get("section_order", [])
        if not isinstance(section_order, list):
            section_order = []

        applicant_name = request.cv_data.get('name', 'Applicant')
        from app.services.latex_renderer import render_cv_tex
        latex_preview = render_cv_tex(request.cv_data, section_order=section_order)

        # Each call gets its own random folder, like an orchestrator session,
        # so the download URL can't be guessed from the applicant's name.
        output_id = str(uuid.uuid4())
        pdf_result = generate_cv_pdf.invoke({
            "cv_json": cv_json,
            "output_filename": request.output_filename,
            "applicant_name": applicant_name,
            "section_order_json": json.dumps(section_order),
            "output_dir": str(get_file_store().session_dir(output_id)),
        })

        if "Error" in pdf_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=pdf_result
            )

        match = re.search(r"The file '([^']+)' is ready for download", pdf_result)
        pdf_path = get_file_store().download_url(output_id, match.group(1)) if match else None

        return GenerateTailoredCVResponse(
            success=True,
            pdf_path=pdf_path,
            latex_preview=latex_preview[:500] + "..." if len(latex_preview) > 500 else latex_preview,
            message=pdf_result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating CV: {str(e)}"
        )


# ============================================
# /generate-cover-letter
# ============================================

@router.post(
    "/generate-cover-letter",
    response_model=GenerateCoverLetterResponse,
    summary="Generate Cover Letter",
    description="Generate a tailored cover-letter PDF from CV + job + optional company info."
)
def generate_cover_letter(request: GenerateCoverLetterRequest):
    """Deterministic cover-letter PDF generation."""
    try:
        try:
            ResumeInfo(**request.cv_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CV data: {str(e)}"
            )

        try:
            JobRequirements(**request.job_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job data: {str(e)}"
            )

        if not is_safe_filename(request.output_filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=UNSAFE_FILENAME_MESSAGE,
            )

        cv_json = json.dumps(request.cv_data)
        job_json = json.dumps(request.job_data)
        company_json = json.dumps(request.company_data) if request.company_data else ""

        content_json = generate_cover_letter_content.invoke({
            "cv_json": cv_json,
            "job_json": job_json,
            "company_json": company_json,
        })

        content_dict = json.loads(content_json)

        applicant_name = request.cv_data.get('name', 'Applicant')
        applicant_email = request.cv_data.get('email', '')
        applicant_phone = request.cv_data.get('phone', '')
        applicant_contact = f"{applicant_email} | {applicant_phone}"

        recipient_info = (
            (request.recipient_info or "").strip()
            or extract_recipient_name(request.job_data)
        )
        if not recipient_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "recipient_info is required when the job posting does not name "
                    "a contact person."
                ),
            )

        output_id = str(uuid.uuid4())
        pdf_result = generate_cover_letter_pdf.invoke({
            "content_json": content_json,
            "output_filename": request.output_filename,
            "applicant_name": applicant_name,
            "applicant_contact": applicant_contact,
            "recipient_info": recipient_info,
            "cv_json": cv_json,
            "job_json": job_json,
            "output_dir": str(get_file_store().session_dir(output_id)),
        })

        if "Error" in pdf_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=pdf_result
            )

        match = re.search(r"The file '([^']+)' is ready for download", pdf_result)
        pdf_path = get_file_store().download_url(output_id, match.group(1)) if match else None

        return GenerateCoverLetterResponse(
            success=True,
            pdf_path=pdf_path,
            content=CoverLetterContent(**content_dict),
            message=pdf_result
        )

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse cover letter content: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating cover letter: {str(e)}"
        )


# ============================================
# /health
# ============================================

@router.get(
    "/health",
    summary="Writer Agent Health Check",
    description="Check if writer agent tools are available."
)
async def writer_health():
    """Health check for the writer-agent tool surface."""
    return {
        "status": "healthy",
        "agent": "writer",
        "available_tools": [
            "analyze_cv_job_alignment",
            "generate_cover_letter_content",
            "generate_cv_pdf",
            "generate_cover_letter_pdf",
        ],
        "message": (
            "Writer agent tool wrappers are operational. The conversational "
            "chat surface lives at /api/orchestrator/* now."
        ),
    }
