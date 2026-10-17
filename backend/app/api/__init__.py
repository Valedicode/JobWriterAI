"""
API Routes

This module exports all API routers for registration in the main application.
"""

from app.api.cv import router as cv_router
from app.api.job import router as job_router
from app.api.writer import router as writer_router
from app.api.audio import router as audio_router
from app.api.orchestrator import router as orchestrator_router
from app.api.documents import router as documents_router

__all__ = [
    "cv_router",
    "job_router",
    "writer_router",
    "audio_router",
    "orchestrator_router",
    "documents_router",
]

