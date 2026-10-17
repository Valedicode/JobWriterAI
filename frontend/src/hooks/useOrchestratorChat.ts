/**
 * useOrchestratorChat - drives the LangGraph orchestrator from the UI.
 *
 *  - Sessions are created via POST /api/orchestrator/start.
 *  - The graph pauses on structured gates; the hook exposes the latest
 *    `pendingGate` so the UI can render an ApprovalGate or ChoiceGate.
 *  - `submitGateResolution(action, opts)` resumes the graph through the
 *    corresponding REST call.
 *  - Free-text chat (`handleSendMessage`) goes to the backend's chat agent,
 *    which answers, answers the pending gate, changes one rewritten bullet,
 *    or proposes a re-run / profile update. A proposal comes back as a
 *    `confirm_change` gate; nothing changes until the user confirms it.
 *  - `runDocumentSave(echo, save)` runs a save from the document editor the
 *    same way: the user's action shows in the transcript, the chat waits while
 *    the PDF is generated again, and the session's reply lands in the chat.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  getErrorMessage,
  orchestratorMessage,
  orchestratorStart,
} from '@/lib/api';
import type {
  BulletDecision,
  DocumentSaveResponse,
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
    opts?: {
      feedback?: string;
      choice?: string;
      decisions?: Record<string, BulletDecision>;
      answers?: Record<string, string>;
    }
  ) => Promise<void>;
  /** Run an editor save; resolves with its result, or rethrows its error after reporting it in the chat. */
  runDocumentSave: (
    echo: string,
    save: (sessionId: string) => Promise<DocumentSaveResponse>
  ) => Promise<DocumentSaveResponse | null>;
  clearError: () => void;
  resetSession: () => void;
}

function flowFromMode(
  flowMode: 'cv_only' | 'job_tailoring' | null | undefined,
  jobData: JobRequirements | null
): OrchestratorFlow {
  if (flowMode === 'cv_only') return 'cv_review';
  if (flowMode === 'job_tailoring') return 'job_tailoring';
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
  const startInFlight = useRef(false);

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
  //
  // `narrations` carries every step narration produced while resolving this
  // request (job summary, score, strategy, ...), not just the text attached
  // to whichever gate the run stopped on. Each entry becomes its own chat
  // bubble, in order, so multi-step turns no longer read as a single
  // truncated reply. Older/malformed responses without `narrations` fall
  // back to the single `narration` field.
  const applyResponse = useCallback((response: OrchestratorResponse) => {
    setSessionId(response.session_id);
    const narrations =
      response.narrations && response.narrations.length > 0
        ? response.narrations
        : response.narration
          ? [response.narration]
          : [];
    if (narrations.length > 0) {
      setMessages((prev) => [
        ...prev,
        ...narrations.map((content, i) => ({
          id: `assistant-${Date.now()}-${prev.length}-${i}`,
          role: 'assistant' as const,
          content,
          timestamp: new Date(),
          generatedFiles:
            i === narrations.length - 1 &&
            response.generated_files &&
            response.generated_files.length > 0
              ? response.generated_files
              : null,
        })),
      ]);
    }
    setPendingGate(response.pending_gate ?? null);
    setDone(Boolean(response.done));
    setGeneratedFiles(response.generated_files || []);
  }, []);

  const initializeSession = useCallback(async () => {
    if (sessionId || !cvData || startInFlight.current) return;

    const flow = flowFromMode(flowMode, jobData);
    if (flow === 'job_tailoring' && !jobData) {
      // Wait until job extraction finishes; calling /start now 400s.
      return;
    }

    startInFlight.current = true;
    setIsInitializing(true);
    setSessionError(null);

    try {
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
      startInFlight.current = false;
      setIsInitializing(false);
    }
  }, [sessionId, cvData, jobData, flowMode, applyResponse]);

  const submitGateResolution = useCallback(
    async (
      action: OrchestratorGateAction,
      opts: {
        feedback?: string;
        choice?: string;
        decisions?: Record<string, BulletDecision>;
        answers?: Record<string, string>;
      } = {}
    ) => {
      if (!sessionId || isLoading) return;

      const hasAnswers = Boolean(opts.answers && Object.keys(opts.answers).length > 0);
      const resolution: OrchestratorGateResolution = {
        action,
        // Tie the answer to the gate on screen, so a double-submit or a stale
        // view can't resolve whichever gate the session has moved on to.
        ...(pendingGate ? { step: pendingGate.step } : {}),
        ...(opts.feedback ? { feedback: opts.feedback } : {}),
        ...(opts.choice ? { choice: opts.choice } : {}),
        ...(opts.decisions && Object.keys(opts.decisions).length > 0
          ? { decisions: opts.decisions }
          : {}),
        ...(hasAnswers ? { answers: opts.answers } : {}),
      };

      // Echo a user-side message describing the action so the transcript
      // shows what was decided.
      const isQuestionGate = pendingGate?.step === 'confirm_eligibility';
      const isConfirmGate = pendingGate?.step === 'confirm_change';
      const echoText =
        action === 'approve'
          ? isQuestionGate
            ? 'Skipped.'
            : isConfirmGate
              ? 'Confirmed.'
              : 'Approved.'
          : action === 'reject'
            ? isConfirmGate
              ? 'Cancelled.'
              : 'Rejected.'
            : action === 'edit'
              ? hasAnswers
                ? `Answered: ${opts.feedback || ''}`
                : `Edit: ${opts.feedback || ''}`
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
    [sessionId, isLoading, pendingGate, applyResponse]
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

  const runDocumentSave = useCallback(
    async (echo: string, save: (sessionId: string) => Promise<DocumentSaveResponse>) => {
      if (!sessionId || isLoading) return null;
      setMessages((prev) => [
        ...prev,
        { id: `user-${Date.now()}`, role: 'user', content: echo, timestamp: new Date() },
      ]);
      setIsLoading(true);
      setChatError(null);
      try {
        const result = await save(sessionId);
        applyResponse(result.session);
        return result;
      } catch (err) {
        const errorMessage = getErrorMessage(err);
        setChatError(errorMessage);
        setMessages((prev) => [
          ...prev,
          {
            id: `error-${Date.now()}`,
            role: 'assistant',
            content: `Sorry, that didn't work: ${errorMessage}`,
            timestamp: new Date(),
          },
        ]);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isLoading, applyResponse]
  );

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
    startInFlight.current = false;
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
    runDocumentSave,
    clearError,
    resetSession,
  };
};
