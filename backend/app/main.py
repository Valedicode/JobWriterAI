"""
FastAPI application entry point
"""

import logging
from datetime import timedelta

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.api import (
    audio_router,
    cv_router,
    job_router,
    orchestrator_router,
    writer_router,
)
from app.config import get_settings
from app.services.generic_session_manager import get_session_manager
from app.services.graph_checkpointer import (
    configure_checkpointer,
    expire_threads,
    get_checkpointer,
    postgres_checkpointer,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    With DATABASE_URL set, orchestrator sessions are stored in Postgres and
    survive restarts; a configured database that can't be reached stops
    startup. Without it they live in memory only.
    """
    print("\n" + "="*70)
    print("Starting JobWriterAI API...")
    print("="*70 + "\n")

    settings = get_settings()
    close_checkpointer = None
    if settings.database_url:
        saver, close_checkpointer = postgres_checkpointer(settings.database_url)
        configure_checkpointer(saver)
        logger.info("Orchestrator sessions are stored in Postgres.")
    else:
        logger.warning("DATABASE_URL is not set: orchestrator sessions live in memory and are lost on restart.")

    expired = expire_threads(get_checkpointer(), timedelta(hours=settings.session_ttl_hours))
    if expired:
        logger.info("Deleted %d expired session(s) at startup.", len(expired))

    yield

    print("\nShutting down JobWriterAI API...")
    if close_checkpointer is not None:
        close_checkpointer()


app = FastAPI(
    title="JobWriterAI API",
    description="AI-powered resume and cover-letter optimization with human-in-the-loop feedback",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS is handled by the frontend's rewrite proxy
# The Next.js development server (next.config.js) proxies API requests
# to this backend, so all requests appear to come from the same origin.
# This eliminates the need for CORS middleware in development and production.
#
# If you need to enable CORS for direct API access (e.g., testing with Postman),
# uncomment the following:
#
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Register API routers
app.include_router(cv_router)
app.include_router(job_router)
app.include_router(writer_router)
app.include_router(audio_router)
app.include_router(orchestrator_router)

# Configure file serving directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "JobWriterAI API",
        "status": "running",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "agents": {
            "cv": "/api/cv",
            "job": "/api/job",
            "writer": "/api/writer",
            "audio": "/api/audio"
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    session_manager = get_session_manager()
    
    return {
        "status": "healthy",
        "version": "0.1.0",
        "agents": {
            "cv_agent": "operational",
            "job_agent": "operational",
            "writer_agent": "operational"
        },
        "active_sessions": session_manager.get_session_count(),
        "endpoints": {
            "cv": "/api/cv",
            "job": "/api/job",
            "writer": "/api/writer",
            "audio": "/api/audio"
        }
    }


@app.get(
    "/api/files/{filename}",
    response_class=FileResponse,
    summary="Download Generated File",
    description="Download generated PDF files (CVs, cover letters) from the server."
)
async def download_file(filename: str):
    """
    Download a generated file.
    
    **Purpose:**
    Serves generated PDF files (CVs, cover letters) to the frontend for download.
    
    **Security:**
    - Validates filename to prevent directory traversal attacks
    - Only serves files from the data directory
    - Returns 404 if file doesn't exist
    
    **Parameters:**
    - `filename`: Name of the file to download (e.g., "john_doe_cv.pdf")
    
    **Returns:**
    - PDF file with appropriate content-type headers
    
    **Usage:**
    ```javascript
    // Frontend example
    const response = await fetch(`/api/files/${filename}`);
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    ```
    """
    try:
        # Security: Validate filename (prevent directory traversal)
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename. Filename cannot contain path separators or '..'."
            )
        
        # Security: Only allow PDF files
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files can be downloaded."
            )
        
        # Construct file path
        file_path = DATA_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {filename}"
            )
        
        # Security: Verify file is actually in DATA_DIR (resolve symlinks)
        try:
            resolved_path = file_path.resolve()
            if not str(resolved_path).startswith(str(DATA_DIR.resolve())):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied."
                )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied."
            )
        
        media_type = "application/pdf"
        
        # Return file
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)