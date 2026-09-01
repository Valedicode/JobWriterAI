/**
 * useOrchestratorChat - drives the LangGraph orchestrator from the UI.
 *
 * Differences from `useWriterChat`:
 *  - Sessions are created via POST /api/orchestrator/start, not /writer/chat/start.
 *  - The graph pauses on structured gates; the hook exposes the latest
 *    `pendingGate` so the UI can render an ApprovalGate or ChoiceGate.
 *  - `submitGateResolution(action, opts)` resumes the graph through the
 *    corresponding REST call. Free-text chat (`handleSendMessage`) is folded
 *    in by the backend as 'edit' feedback for the currently pending gate when
 *    the gate allows edits; otherwise the backend replies with the set of
 *    actions that are actually available.
 *
 * Shape matches useWriterChat closely so `page.tsx` can swap one for the other
 * behind a feature flag without UI churn.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  getErrorMessage,
  orchestratorMessage,
  orchestratorStart,
} from '@/lib/api';
import type {
  DownloadableFile,
  JobRequirements,
  Message,
  OrchestratorFlow,
  OrchestratorGateAction,
  OrchestratorGatePayload,
  OrchestratorGateResolution,
  OrchestratorResponse,
  ResumeInfo,
} from '@/types';

interface UseOrchestratorChatProps {
  cvData: ResumeInfo | null;
  jobData: JobRequirements | null;
  flowMode?: 'cv_only' | 'job_tailoring' | null;
}

interface UseOrchestratorChatReturn {
  // Session state
  sessionId: string | null;
  isInitializing: boolean;
  sessionError: string | null;

  // Chat state
  messages: Message[];
  fadingOutMessageId: string | null;
  inputText: string;
  setInputText: (text: string) => void;
  isLoading: boolean;
  chatError: string | null;

  // Gate state
  pendingGate: OrchestratorGatePayload | null;
  done: boolean;
  generatedFiles: DownloadableFile[];

  // Refs (same names as useWriterChat for drop-in swap)
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;

  // Methods
  initializeSession: () => Promise<void>;
  handleSendMessage: () => Promise<void>;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  submitGateResolution: (
    action: OrchestratorGateAction,
    opts?: { feedback?: string; choice?: string }
  ) => Promise<void>;
  clearError: () => void;
  resetSession: () => void;
}

function flowFromMode(
  flowMode: 'cv_only' | 'job_tailoring' | null | undefined,
  jobData: JobRequirements | null
): OrchestratorFlow {
  if (flowMode === 'cv_only') return 'cv_review';
  if (flowMode === 'job_tailoring') return 'job_tailoring';
  // No explicit choice -> infer from presence of job data.
  return jobData ? 'job_tailoring' : 'cv_review';
}

export const useOrchestratorChat = ({
  cvData,
  jobData,
  flowMode,
}: UseOrchestratorChatProps): UseOrchestratorChatReturn => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [fadingOutMessageId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const [pendingGate, setPendingGate] = useState<OrchestratorGatePayload | null>(null);
  const [done, setDone] = useState(false);
  const [generatedFiles, setGeneratedFiles] = useState<DownloadableFile[]>([]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const reduce =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    messagesEndRef.current?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputText]);

  // Helper: fold an OrchestratorResponse into local state.
  const applyResponse = useCallback((response: OrchestratorResponse) => {
    setSessionId(response.session_id);
    if (response.narration) {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}-${prev.length}`,
          role: 'assistant',
          content: response.narration,
          timestamp: new Date(),
          generatedFiles:
            response.generated_files && response.generated_files.length > 0
              ? response.generated_files
              : null,
        },
      ]);
    }
    setPendingGate(response.pending_gate ?? null);
    setDone(Boolean(response.done));
    setGeneratedFiles(response.generated_files || []);
  }, []);

  const initializeSession = useCallback(async () => {
    if (sessionId || !cvData) return;

    setIsInitializing(true);
    setSessionError(null);

    try {
      const flow = flowFromMode(flowMode, jobData);
      const response = await orchestratorStart({
        flow,
        cv_data: cvData,
        job_data: jobData || undefined,
      });
      if (response.success) {
        applyResponse(response);
      } else {
        setSessionError('Failed to start orchestrator session');
      }
    } catch (err) {
      setSessionError(getErrorMessage(err));
      console.error('Orchestrator init error:', err);
    } finally {
      setIsInitializing(false);
    }
  }, [sessionId, cvData, jobData, flowMode, applyResponse]);

  const submitGateResolution = useCallback(
    async (
      action: OrchestratorGateAction,
      opts: { feedback?: string; choice?: string } = {}
    ) => {
      if (!sessionId || isLoading) return;

      const resolution: OrchestratorGateResolution = {
        action,
        ...(opts.feedback ? { feedback: opts.feedback } : {}),
        ...(opts.choice ? { choice: opts.choice } : {}),
      };

      // Echo a user-side message describing the action so the transcript
      // shows what was decided.
      const echoText =
        action === 'approve'
          ? 'Approved.'
          : action === 'reject'
            ? 'Rejected.'
            : action === 'edit'
              ? `Edit: ${opts.feedback || ''}`
              : `Selected: ${opts.choice || ''}`;
      setMessages((prev) => [
        ...prev,
        {
          id: `user-${Date.now()}`,
          role: 'user',
          content: echoText,
          timestamp: new Date(),
        },
      ]);

      setIsLoading(true);
      setChatError(null);
      try {
        const response = await orchestratorMessage({
          session_id: sessionId,
          kind: 'gate_resolution',
          resolution,
        });
        applyResponse(response);
      } catch (err) {
        const errorMessage = getErrorMessage(err);
        setChatError(errorMessage);
        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}`,
            role: 'assistant',
            content: `Sorry, that didn't work: ${errorMessage}. Please try again.`,
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isLoading, applyResponse]
  );

  const handleSendMessage = useCallback(async () => {
    if (!inputText.trim() || isLoading || !sessionId) return;

    const text = inputText.trim();
    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: new Date(),
      },
    ]);
    setInputText('');
    setIsLoading(true);
    setChatError(null);

    try {
      const response = await orchestratorMessage({
        session_id: sessionId,
        kind: 'chat',
        text,
      });
      applyResponse(response);
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setChatError(errorMessage);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Sorry, I encountered an error: ${errorMessage}. Please try again.`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, isLoading, sessionId, applyResponse]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    },
    [handleSendMessage]
  );

  const clearError = useCallback(() => {
    setChatError(null);
    setSessionError(null);
  }, []);

  const resetSession = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setInputText('');
    setSessionError(null);
    setChatError(null);
    setPendingGate(null);
    setDone(false);
    setGeneratedFiles([]);
  }, []);

  return {
    sessionId,
    isInitializing,
    sessionError,
    messages,
    fadingOutMessageId,
    inputText,
    setInputText,
    isLoading,
    chatError,
    pendingGate,
    done,
    generatedFiles,
    textareaRef,
    messagesEndRef,
    initializeSession,
    handleSendMessage,
    handleKeyDown,
    submitGateResolution,
    clearError,
    resetSession,
  };
};
