"""
Audio Transcription API Endpoints

Provides endpoints for speech-to-text transcription and translation using OpenAI's Audio API.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Query
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from typing import Optional, List
import tempfile
import os

from app.models.schemas import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
    ErrorResponse
)
from app.config import get_settings
from openai import OpenAI

router = APIRouter(prefix="/api/audio", tags=["Audio Transcription"])

# Supported audio file formats
SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file."""
    # Check file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # Check content type (optional, as browsers may not always send correct MIME types)
    if file.content_type:
        valid_mime_types = [
            'audio/mpeg', 'audio/mp3', 'audio/mp4', 'audio/mpeg',
            'audio/mpga', 'audio/x-m4a', 'audio/wav', 'audio/webm',
            'video/mp4', 'video/webm'  # Some formats can be in video containers
        ]
        if not any(mime in file.content_type for mime in valid_mime_types):
            # Don't fail on content type, just log a warning
            pass


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe Audio to Text",
    description="Transcribe audio file to text using OpenAI's Audio API. Supports multiple models and formats."
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="gpt-4o-transcribe", description="Model to use"),
    response_format: str = Form(default="json", description="Response format"),
    language: Optional[str] = Form(default=None, description="Language code (ISO 639-1 or 639-3)"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt to guide transcription"),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature (0-1)"),
    timestamp_granularities: Optional[List[str]] = Form(default=None, description="Timestamp granularities (word, segment)"),
    chunking_strategy: Optional[str] = Form(default=None, description="Chunking strategy for long audio")
):
    """
    Transcribe audio file to text.
    
    **Supported Models:**
    - `whisper-1`: Original Whisper model (supports all formats including srt, vtt)
    - `gpt-4o-transcribe`: Higher quality model (supports json, text)
    - `gpt-4o-mini-transcribe`: Faster, lower cost model (supports json, text)
    - `gpt-4o-transcribe-diarize`: Speaker-aware transcription (requires chunking_strategy for audio > 30s)
    
    **Supported Formats:**
    - Input: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - Output: json, text, srt, verbose_json, vtt, diarized_json
    
    **File Size Limit:** 25 MB
    
    **Parameters:**
    - `model`: Model to use for transcription
    - `response_format`: Format of the response (json, text, srt, verbose_json, vtt, diarized_json)
    - `language`: Optional language code to help with transcription accuracy
    - `prompt`: Optional text to guide the model (e.g., correct spellings, context)
    - `temperature`: Sampling temperature (0-1), higher = more random
    - `timestamp_granularities`: For whisper-1 only, can include "word" and/or "segment"
    - `chunking_strategy`: Required for gpt-4o-transcribe-diarize when audio > 30s (use "auto")
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/audio/transcribe" \\
      -F "file=@audio.mp3" \\
      -F "model=gpt-4o-transcribe" \\
      -F "response_format=text" \\
      -F "prompt=This is a lecture about AI and machine learning."
    ```
    """
    try:
        # Validate file
        validate_audio_file(file)
        
        # Read file into memory
        audio_bytes = await file.read()
        
        # Validate file size
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(audio_bytes) / (1024*1024):.2f}MB) exceeds maximum allowed size of 25MB"
            )
        
        # Validate model and response_format combination
        if model == "gpt-4o-transcribe-diarize":
            if response_format not in ["json", "text", "diarized_json"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model gpt-4o-transcribe-diarize only supports json, text, or diarized_json formats"
                )
            # Check if chunking_strategy is needed (for audio > 30s, we'd need to check duration, but we'll let OpenAI handle it)
            if chunking_strategy is None:
                # Warn but don't fail - OpenAI will handle it
                pass
        
        if model in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            if response_format not in ["json", "text"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model {model} only supports json or text formats"
                )
        
        if timestamp_granularities and model != "whisper-1":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="timestamp_granularities is only supported for whisper-1 model"
            )
        
        # Get OpenAI client
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Create temporary file for OpenAI SDK
        # OpenAI SDK expects a file-like object, so we'll use a temporary file
        tmp_file_path = None
        file_handle = None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                tmp_file_path = tmp_file.name
            
            # Open file for OpenAI SDK (outside the with block to ensure previous handle is closed)
            file_handle = open(tmp_file_path, "rb")
            
            # Prepare transcription parameters
            transcription_params = {
                "file": file_handle,
                "model": model,
                "response_format": response_format,
            }
            
            # Add optional parameters
            if language:
                transcription_params["language"] = language
            if prompt:
                transcription_params["prompt"] = prompt
            if temperature is not None:
                transcription_params["temperature"] = temperature
            if timestamp_granularities:
                transcription_params["timestamp_granularities"] = timestamp_granularities
            if chunking_strategy:
                transcription_params["chunking_strategy"] = chunking_strategy
            
            # Call OpenAI Audio API
            # Blocking HTTP call: keep it off the event loop.
            transcription = await run_in_threadpool(client.audio.transcriptions.create, **transcription_params)
            
            # Parse response based on format
            if response_format == "text":
                return TranscriptionResponse(
                    success=True,
                    text=transcription,
                    message="Transcription completed successfully"
                )
            elif response_format in ["json", "verbose_json", "diarized_json"]:
                # OpenAI returns different structures based on format
                if hasattr(transcription, 'text'):
                    text = transcription.text
                else:
                    text = str(transcription)
                
                # Extract segments and words if available
                segments = None
                words = None
                
                if hasattr(transcription, 'segments'):
                    segments = [segment.model_dump() if hasattr(segment, 'model_dump') else segment for segment in transcription.segments]
                
                if hasattr(transcription, 'words'):
                    words = [word.model_dump() if hasattr(word, 'model_dump') else word for word in transcription.words]
                
                return TranscriptionResponse(
                    success=True,
                    text=text,
                    segments=segments,
                    words=words,
                    message="Transcription completed successfully"
                )
            else:
                # For srt, vtt formats, return as text
                return TranscriptionResponse(
                    success=True,
                    text=str(transcription),
                    message="Transcription completed successfully"
                )
        
        finally:
            # Close file handle first
            if file_handle is not None:
                try:
                    file_handle.close()
                except Exception:
                    pass
            
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    # Log but don't fail if cleanup fails
                    print(f"Warning: Could not delete temporary file {tmp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}"
        )


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate Audio to English",
    description="Translate audio file to English text using OpenAI's Audio API. Only supports whisper-1 model."
)
async def translate_audio(
    file: UploadFile = File(..., description="Audio file to translate"),
    model: str = Form(default="whisper-1", description="Model to use (only whisper-1 supported)"),
    response_format: str = Form(default="json", description="Response format"),
    prompt: Optional[str] = Form(default=None, description="Optional prompt to guide translation"),
    temperature: Optional[float] = Form(default=None, description="Sampling temperature (0-1)")
):
    """
    Translate audio file to English text.
    
    **Note:** This endpoint translates the audio to English regardless of the source language.
    Use `/transcribe` if you want to preserve the original language.
    
    **Supported Models:**
    - `whisper-1`: Only model supported for translation
    
    **Supported Formats:**
    - Input: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - Output: json, text, srt, verbose_json, vtt
    
    **File Size Limit:** 25 MB
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/audio/translate" \\
      -F "file=@german_audio.mp3" \\
      -F "model=whisper-1" \\
      -F "response_format=text"
    ```
    """
    try:
        # Validate file
        validate_audio_file(file)
        
        # Validate model
        if model != "whisper-1":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Translation endpoint only supports whisper-1 model"
            )
        
        # Read file into memory
        audio_bytes = await file.read()
        
        # Validate file size
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(audio_bytes) / (1024*1024):.2f}MB) exceeds maximum allowed size of 25MB"
            )
        
        # Get OpenAI client
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        
        # Create temporary file for OpenAI SDK
        tmp_file_path = None
        file_handle = None
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                tmp_file_path = tmp_file.name
            
            # Open file for OpenAI SDK (outside the with block to ensure previous handle is closed)
            file_handle = open(tmp_file_path, "rb")
            
            # Prepare translation parameters
            translation_params = {
                "file": file_handle,
                "model": model,
                "response_format": response_format,
            }
            
            # Add optional parameters
            if prompt:
                translation_params["prompt"] = prompt
            if temperature is not None:
                translation_params["temperature"] = temperature
            
            # Call OpenAI Audio API
            # Blocking HTTP call: keep it off the event loop.
            translation = await run_in_threadpool(client.audio.translations.create, **translation_params)
            
            # Parse response based on format
            if response_format == "text":
                return TranslationResponse(
                    success=True,
                    text=translation,
                    message="Translation completed successfully"
                )
            elif response_format in ["json", "verbose_json"]:
                # OpenAI returns different structures based on format
                if hasattr(translation, 'text'):
                    text = translation.text
                else:
                    text = str(translation)
                
                # Extract segments and words if available
                segments = None
                words = None
                
                if hasattr(translation, 'segments'):
                    segments = [segment.model_dump() if hasattr(segment, 'model_dump') else segment for segment in translation.segments]
                
                if hasattr(translation, 'words'):
                    words = [word.model_dump() if hasattr(word, 'model_dump') else word for word in translation.words]
                
                return TranslationResponse(
                    success=True,
                    text=text,
                    segments=segments,
                    words=words,
                    message="Translation completed successfully"
                )
            else:
                # For srt, vtt formats, return as text
                return TranslationResponse(
                    success=True,
                    text=str(translation),
                    message="Translation completed successfully"
                )
        
        finally:
            # Close file handle first
            if file_handle is not None:
                try:
                    file_handle.close()
                except Exception:
                    pass
            
            # Clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    # Log but don't fail if cleanup fails
                    print(f"Warning: Could not delete temporary file {tmp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating audio: {str(e)}"
        )

