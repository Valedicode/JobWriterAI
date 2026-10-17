"""
FastAPI application entry point
"""

import logging
from datetime import timedelta

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.api import (
    audio_router,
    cv_router,
    documents_router,
    job_router,
    orchestrator_router,
    writer_router,
)
from app.config import get_settings
from app.services.file_store import get_file_store
from app.services.generic_session_manager import get_session_manager
from app.services.graph_checkpointer import configure_checkpointer, postgres_checkpointer
from app.services.session_retention import expire_sessions

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

    expired = expire_sessions(timedelta(hours=settings.session_ttl_hours))
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
app.include_router(documents_router)

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
    "/api/files/{session_id}/{filename}",
    response_class=FileResponse,
    summary="Download Generated File",
    description=(
        "Download a PDF (CV or cover letter) generated in one orchestrator session; "
        "with ?inline=true the browser shows it instead (the editor's preview)."
    ),
)
async def download_file(session_id: str, filename: str, inline: bool = False):
    """
    Serve a generated PDF from its session's folder in the file store.

    The URL is the ``download_url`` the orchestrator returns with each
    generated file. Files are only looked up inside the named session's
    folder, so the session id (a random UUID) is required to fetch one.
    There are no user accounts yet; once they exist (plan item 14) this
    route should also check that the session belongs to the caller.
    """
    path = get_file_store().path_for(session_id, filename)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{"inline" if inline else "attachment"}; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)