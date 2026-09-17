/**
 * API Service Layer for Backend Integration
 * 
 * This module provides functions to interact with the backend API endpoints.
 * All API calls go through the Next.js proxy configured in next.config.ts
 */

import {
  CVExtractionResponse,
  CVClarificationRequest,
  CVClarificationResponse,
  JobExtractionResponse,
  JobURLRequest,
  JobTextRequest,
  CompanyResearchResponse,
  CompanyResearchRequest,
  CVJobAlignmentRequest,
  CVJobAlignmentResponse,
  GenerateTailoredCVRequest,
  GenerateTailoredCVResponse,
  GenerateCoverLetterRequest,
  GenerateCoverLetterResponse,
  OrchestratorStartRequest,
  OrchestratorMessageRequest,
  OrchestratorResponse,
  OrchestratorStateResponse,
  TranscriptionRequest,
  TranscriptionResponse,
  TranslationRequest,
  TranslationResponse,
  ErrorResponse,
} from '@/types';

// ============================================
// API Configuration
// ============================================

const API_BASE_URL = '/api'; // Proxied through Next.js

class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Orchestrator turns can run 20-60s+ (LLM calls, LaTeX rendering). Give the
// request a hard ceiling matching the Next proxy timeout so a hung backend
// surfaces as a clear message instead of a spinner that never resolves.
const DEFAULT_TIMEOUT_MS = 300_000;

/** Map a status code to a message a user can act on. */
function messageForStatus(status: number, fallback: string): string {
  switch (status) {
    case 400:
      return fallback || 'The request was rejected. Please check your input and try again.';
    case 401:
    case 403:
      return 'This session is no longer authorised. Refresh the page to start again.';
    case 404:
      return 'That session or resource was not found. It may have expired — refresh to start again.';
    case 429:
      return 'Too many requests right now. Wait a moment, then try again.';
    default:
      if (status >= 500) return 'The server ran into a problem. Please try again in a moment.';
      return fallback || 'Something went wrong. Please try again.';
  }
}

/**
 * Generic JSON fetch wrapper: timeout, status-aware messages, network fallback.
 */
async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit & { timeoutMs?: number }
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...init } = options ?? {};
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...init,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...init.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const raw = errorData.detail || errorData.error || errorData.message || '';
      throw new APIError(
        messageForStatus(response.status, raw),
        response.status,
        raw || undefined
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof APIError) throw error;
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError(
        'This is taking longer than expected. The request may still be processing — refresh the page to check your session.',
        0,
        'Request timeout'
      );
    }
    if (
      error instanceof TypeError ||
      (error instanceof Error &&
        (error.message.includes('fetch') ||
          error.message.includes('ECONNRESET') ||
          error.message.includes('socket')))
    ) {
      throw new APIError(
        'Cannot reach the server. Check your connection and make sure the backend is running, then try again.',
        0,
        'Network connection failed'
      );
    }
    throw new APIError(error instanceof Error ? error.message : 'Network error', 0);
  } finally {
    clearTimeout(timeoutId);
  }
}

// ============================================
// CV Agent API
// ============================================

/**
 * Upload CV PDF file for extraction
 */
export async function uploadCV(file: File): Promise<CVExtractionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    // Use AbortController for timeout handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

    const response = await fetch(`${API_BASE_URL}/cv/upload`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // Don't set Content-Type header - browser will set it with boundary
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || `CV upload failed with status ${response.status}`;
      throw new APIError(
        errorMessage,
        response.status,
        errorData.detail
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    // AbortController timeout
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError(
        'CV upload timed out. The file may be too large or the server is taking too long to process. Please try again or contact support.',
        0,
        'Request timeout'
      );
    }
    // Network errors or connection reset
    if (error instanceof TypeError || (error instanceof Error && (error.message.includes('fetch') || error.message.includes('ECONNRESET') || error.message.includes('socket')))) {
      throw new APIError(
        'Cannot connect to backend server. Please ensure:\n1. The backend is running at http://localhost:8000\n2. Check backend logs for errors\n3. Try restarting the backend server',
        0,
        'Network connection failed'
      );
    }
    throw new APIError(
      error instanceof Error ? error.message : 'Upload failed',
      0
    );
  }
}

/**
 * Submit clarifications for ambiguous CV data
 */
export async function clarifyCV(
  request: CVClarificationRequest
): Promise<CVClarificationResponse> {
  return apiFetch<CVClarificationResponse>('/cv/clarify', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Validate CV data structure
 */
export async function validateCVData(cvData: any): Promise<any> {
  return apiFetch<any>('/cv/validate', {
    method: 'POST',
    body: JSON.stringify(cvData),
  });
}

// ============================================
// Job Agent API
// ============================================

/**
 * Extract job requirements from URL(s)
 */
export async function extractJobFromURL(
  request: JobURLRequest
): Promise<JobExtractionResponse> {
  return apiFetch<JobExtractionResponse>('/job/extract/url', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Extract job requirements from pasted text
 */
export async function extractJobFromText(
  request: JobTextRequest
): Promise<JobExtractionResponse> {
  return apiFetch<JobExtractionResponse>('/job/extract/text', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Research company information
 */
export async function researchCompany(
  request: CompanyResearchRequest
): Promise<CompanyResearchResponse> {
  return apiFetch<CompanyResearchResponse>('/job/research/company', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// ============================================
// Writer Agent API
// ============================================

/**
 * Analyze CV-job alignment and get tailoring plan
 */
export async function analyzeAlignment(
  request: CVJobAlignmentRequest
): Promise<CVJobAlignmentResponse> {
  return apiFetch<CVJobAlignmentResponse>('/writer/analyze-alignment', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Generate tailored CV PDF
 */
export async function generateTailoredCV(
  request: GenerateTailoredCVRequest
): Promise<GenerateTailoredCVResponse> {
  return apiFetch<GenerateTailoredCVResponse>('/writer/generate-cv', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Generate cover letter PDF
 */
export async function generateCoverLetter(
  request: GenerateCoverLetterRequest
): Promise<GenerateCoverLetterResponse> {
  return apiFetch<GenerateCoverLetterResponse>('/writer/generate-cover-letter', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// Legacy Writer chat helpers (startWriterChat, sendWriterMessage,
// generateResumeSummary) were removed during the orchestrator migration.
// All chat traffic now goes through the orchestrator API below.

// ============================================
// Orchestrator API (LangGraph backend)
// ============================================

/**
 * Start a new orchestrator session. Runs the graph until the first interrupt
 * and returns the pending gate (if any).
 */
export async function orchestratorStart(
  request: OrchestratorStartRequest
): Promise<OrchestratorResponse> {
  return apiFetch<OrchestratorResponse>('/orchestrator/start', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Resume the orchestrator with either a structured gate resolution or a free
 * chat turn. Returns the next narration and the next pending gate.
 */
export async function orchestratorMessage(
  request: OrchestratorMessageRequest
): Promise<OrchestratorResponse> {
  return apiFetch<OrchestratorResponse>('/orchestrator/message', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Debug snapshot of a session's current state. Used for recovery/refresh.
 */
export async function orchestratorState(
  sessionId: string
): Promise<OrchestratorStateResponse> {
  return apiFetch<OrchestratorStateResponse>(`/orchestrator/state/${sessionId}`, {
    method: 'GET',
  });
}

// ============================================
// File Download API
// ============================================

/**
 * Download generated file (CV or cover letter)
 */
export async function downloadFile(filename: string): Promise<Blob> {
  try {
    const response = await fetch(`${API_BASE_URL}/files/${filename}`);

    if (!response.ok) {
      throw new APIError('File download failed', response.status);
    }

    return await response.blob();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      error instanceof Error ? error.message : 'Download failed',
      0
    );
  }
}

/**
 * Helper to trigger browser download
 */
export function triggerFileDownload(blob: Blob, filename: string) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

// ============================================
// Health Check
// ============================================

/**
 * Check API health status
 */
export async function healthCheck(): Promise<any> {
  return apiFetch<any>('/health', {
    method: 'GET',
  });
}

// ============================================
// Audio Transcription API
// ============================================

/**
 * Transcribe audio file to text
 */
export async function transcribeAudio(
  request: TranscriptionRequest
): Promise<TranscriptionResponse> {
  const formData = new FormData();
  formData.append('file', request.file);
  
  if (request.model) {
    formData.append('model', request.model);
  }
  if (request.response_format) {
    formData.append('response_format', request.response_format);
  }
  if (request.language) {
    formData.append('language', request.language);
  }
  if (request.prompt) {
    formData.append('prompt', request.prompt);
  }
  if (request.temperature !== undefined) {
    formData.append('temperature', request.temperature.toString());
  }
  if (request.timestamp_granularities) {
    request.timestamp_granularities.forEach(granularity => {
      formData.append('timestamp_granularities[]', granularity);
    });
  }
  if (request.chunking_strategy) {
    formData.append('chunking_strategy', request.chunking_strategy);
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

    const response = await fetch(`${API_BASE_URL}/audio/transcribe`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // Don't set Content-Type header - browser will set it with boundary
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || `Transcription failed with status ${response.status}`;
      throw new APIError(
        errorMessage,
        response.status,
        errorData.detail
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError(
        'Transcription timed out. The file may be too large or the server is taking too long to process. Please try again.',
        0,
        'Request timeout'
      );
    }
    if (error instanceof TypeError || (error instanceof Error && (error.message.includes('fetch') || error.message.includes('ECONNRESET')))) {
      throw new APIError(
        'Cannot connect to backend server. Please ensure the backend is running at http://localhost:8000',
        0,
        'Network connection failed'
      );
    }
    throw new APIError(
      error instanceof Error ? error.message : 'Transcription failed',
      0
    );
  }
}

/**
 * Translate audio file to English
 */
export async function translateAudio(
  request: TranslationRequest
): Promise<TranslationResponse> {
  const formData = new FormData();
  formData.append('file', request.file);
  
  if (request.model) {
    formData.append('model', request.model);
  }
  if (request.response_format) {
    formData.append('response_format', request.response_format);
  }
  if (request.prompt) {
    formData.append('prompt', request.prompt);
  }
  if (request.temperature !== undefined) {
    formData.append('temperature', request.temperature.toString());
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

    const response = await fetch(`${API_BASE_URL}/audio/translate`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // Don't set Content-Type header - browser will set it with boundary
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || `Translation failed with status ${response.status}`;
      throw new APIError(
        errorMessage,
        response.status,
        errorData.detail
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new APIError(
        'Translation timed out. The file may be too large or the server is taking too long to process. Please try again.',
        0,
        'Request timeout'
      );
    }
    if (error instanceof TypeError || (error instanceof Error && (error.message.includes('fetch') || error.message.includes('ECONNRESET')))) {
      throw new APIError(
        'Cannot connect to backend server. Please ensure the backend is running at http://localhost:8000',
        0,
        'Network connection failed'
      );
    }
    throw new APIError(
      error instanceof Error ? error.message : 'Translation failed',
      0
    );
  }
}

// ============================================
// Error Handling Utilities
// ============================================

export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError;
}

export function getErrorMessage(error: unknown): string {
  if (isAPIError(error)) {
    // `message` is the user-facing string; `detail` is the raw backend text.
    return error.message || error.detail || 'Something went wrong. Please try again.';
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
}

export { APIError };

