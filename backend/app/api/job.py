"""
Job Agent API Endpoints

Provides endpoints for job posting extraction (URL/text) and company research.
"""

from fastapi import APIRouter, HTTPException, status
import json
import logging

from app.models.schemas import (
    JobURLRequest,
    JobTextRequest,
    JobExtractionResponse,
    CompanyResearchRequest,
    CompanyResearchResponse,
    JobRequirements,
    CompanyInfo
)
from app.agents.job_agent import (
    retrieve_web,
    retrieve_text,
    collect_company_info
)

router = APIRouter(prefix="/api/job", tags=["Job Agent"])
logger = logging.getLogger(__name__)


@router.post(
    "/extract/url",
    response_model=JobExtractionResponse,
    summary="Extract Job from URL",
    description="Extract structured job requirements from one or more job posting URLs."
)
def extract_job_from_url(request: JobURLRequest):
    """
    Extract job requirements from URL(s).
    
    **Workflow:**
    1. Scrapes the web page(s)
    2. Extracts structured job information using LLM
    3. Returns requirements, skills, responsibilities, etc.
    
    **Parameters:**
    - `urls`: List of job posting URLs (can be single URL in a list)
    
    **Returns:**
    - Structured job requirements data
    
    **Example:**
    ```json
    {
        "urls": ["https://company.com/careers/software-engineer"]
    }
    ```
    """
    try:
        if not request.urls:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one URL must be provided"
            )
        
        # Validate URLs
        for url in request.urls:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid URL format: {url}"
                )
        
        # Extract job information from URL(s)
        job_data_json = retrieve_web.invoke({"web_urls": request.urls})
        job_data_dict = json.loads(job_data_json)
        
        return JobExtractionResponse(
            success=True,
            job_data=JobRequirements(**job_data_dict),
            message=f"Job posting extracted successfully from {len(request.urls)} URL(s)."
        )
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse job data: {str(e)}"
        )
    except Exception as e:
        logger.exception("Job extract/url failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting job from URL: {str(e)}"
        )


@router.post(
    "/extract/text",
    response_model=JobExtractionResponse,
    summary="Extract Job from Text",
    description="Extract structured job requirements from pasted job description text."
)
def extract_job_from_text(request: JobTextRequest):
    """
    Extract job requirements from raw text.
    
    **Use this when:**
    - User doesn't have a URL
    - User copies and pastes job description directly
    - Job posting is from email, PDF, or internal system
    
    **Workflow:**
    1. Receives raw job posting text
    2. Uses LLM to extract structured information
    3. Returns requirements, skills, responsibilities, etc.
    
    **Parameters:**
    - `job_text`: Raw job posting text
    
    **Returns:**
    - Structured job requirements data
    """
    try:
        if not request.job_text or len(request.job_text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job text must be at least 50 characters long"
            )
        
        # Extract job information from text
        job_data_json = retrieve_text.invoke({"info": request.job_text})
        job_data_dict = json.loads(job_data_json)
        
        return JobExtractionResponse(
            success=True,
            job_data=JobRequirements(**job_data_dict),
            message="Job posting extracted successfully from text."
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse job data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting job from text: {str(e)}"
        )


@router.post(
    "/research/company",
    response_model=CompanyResearchResponse,
    summary="Research Company Information",
    description="Search the web for company information including values, culture, mission, and recent news."
)
def research_company(request: CompanyResearchRequest):
    """
    Research company information using web search.
    
    **Purpose:**
    - Helps tailor cover letters with company-specific insights
    - Provides context about company culture and values
    - Identifies recent news and developments
    
    **Workflow:**
    1. Searches the web using Tavily API
    2. Extracts structured company information
    3. Returns values, culture, mission, news, etc.
    
    **Parameters:**
    - `company_name`: Name of the company to research
    
    **Returns:**
    - Company info: name, industry, size, values, culture, news, products
    
    **Note:** This endpoint requires Tavily API key in environment variables.
    """
    try:
        if not request.company_name or len(request.company_name.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company name must be at least 2 characters long"
            )
        
        # Research company information
        company_data_json = collect_company_info.invoke({
            "company_name": request.company_name
        })
        company_data_dict = json.loads(company_data_json)
        
        return CompanyResearchResponse(
            success=True,
            company_data=CompanyInfo(**company_data_dict),
            message=f"Company research completed for {request.company_name}."
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse company data: {str(e)}"
        )
    except Exception as e:
        # Check if it's a Tavily API error
        error_message = str(e)
        if "tavily" in error_message.lower() or "api" in error_message.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Company research service unavailable: {error_message}"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error researching company: {error_message}"
        )


@router.post(
    "/validate",
    response_model=dict,
    summary="Validate Job Data",
    description="Validate job data structure without extraction (useful for testing)."
)
async def validate_job_data(job_data: JobRequirements):
    """
    Validate job requirements data structure.
    
    This is a utility endpoint for testing and validation.
    """
    return {
        "success": True,
        "message": "Job data is valid",
        "job_data": job_data.model_dump()
    }


