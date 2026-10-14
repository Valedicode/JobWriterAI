import { useState } from 'react';
import { BulletDecision, Message, OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { EmptyState } from './EmptyState';
import { ApprovalGate } from './ApprovalGate';
import { ChoiceGate } from './ChoiceGate';
import { InputGate } from './InputGate';
import { EligibilityGate } from './EligibilityGate';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ReviewPanel } from './ReviewPanel';

interface ChatContainerProps {
  messages: Message[];
  fadingOutMessageId?: string | null;
  inputText: string;
  isLoading: boolean;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  onInputChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSendMessage: () => void;
  onClickUpload: () => void;
  sessionReady?: boolean;
  // Orchestrator-only: when set, render the structured gate panel above the
  // free-text input. Legacy writer chat leaves these unset and the UI behaves
  // exactly as before.
  pendingGate?: OrchestratorGatePayload | null;
  onSubmitGateResolution?: (
    action: OrchestratorGateAction,
    opts?: {
      feedback?: string;
      choice?: string;
      decisions?: Record<string, BulletDecision>;
      answers?: Record<string, string>;
    }
  ) => void | Promise<void>;
}

export const ChatContainer = ({
  messages,
  fadingOutMessageId,
  inputText,
  isLoading,
  textareaRef,
  messagesEndRef,
  onInputChange,
  onKeyDown,
  onSendMessage,
  onClickUpload,
  sessionReady = true,
  pendingGate,
  onSubmitGateResolution,
}: ChatContainerProps) => {
  // Gate steps shown so far, for the review panel's step list. Updated while
  // rendering (React's "adjust state on prop change" pattern), not in an effect.
  const [shownSteps, setShownSteps] = useState<string[]>([]);
  if (pendingGate && !shownSteps.includes(pendingGate.step)) {
    setShownSteps([...shownSteps, pendingGate.step]);
  }
  const doneSteps = shownSteps.filter((s) => s !== pendingGate?.step);
  const files = [
    ...new Map(messages.flatMap((m) => m.generatedFiles ?? []).map((f) => [f.download_url, f])).values(),
  ];

  if (messages.length === 0) {
    return (
      <EmptyState
        textareaRef={textareaRef}
        inputText={inputText}
        isLoading={isLoading}
        onInputChange={onInputChange}
        onKeyDown={onKeyDown}
        onSendMessage={onSendMessage}
        onClickUpload={onClickUpload}
      />
    );
  }

  return (
    // Below lg: one column (messages, gate, input). From lg: chat on the left,
    // review panel on the right, each scrolling on its own.
    <div className="flex flex-1 flex-col rounded-2xl border border-border bg-surface shadow-sm animate-fade-in lg:grid lg:h-[calc(100dvh-9rem)] lg:min-h-[34rem] lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:overflow-hidden">
      <div className="contents lg:flex lg:min-h-0 lg:flex-col lg:border-r lg:border-border">
      {/* Chat Messages Area */}
      <div className="order-1 flex-1 overflow-y-auto p-6 max-lg:min-h-0 lg:min-h-0">
        <div className="mx-auto max-w-3xl">
          <div className="space-y-4" role="log" aria-live="polite" aria-label="Conversation">
            {messages.map((message) => (
              <ErrorBoundary
                key={message.id}
                fallback={
                  <p className="text-sm text-ink-muted">
                    (A message couldn’t be displayed.)
                  </p>
                }
              >
                <ChatMessage
                  message={message}
                  generatedFiles={message.generatedFiles}
                  isFadingOut={fadingOutMessageId === message.id}
                />
              </ErrorBoundary>
            ))}
            
            {/* Loading indicator */}
            {isLoading && (
              <div className="flex gap-4 justify-start" role="status" aria-label="Assistant is responding">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-fg">
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div className="rounded-2xl bg-surface-sunken px-4 py-3">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-ink-faint [animation-delay:-0.3s]"></div>
                    <div className="h-2 w-2 animate-pulse rounded-full bg-ink-faint [animation-delay:-0.15s]"></div>
                    <div className="h-2 w-2 animate-pulse rounded-full bg-ink-faint"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="order-3">
      <ChatInput
        textareaRef={textareaRef}
        inputText={inputText}
        isLoading={isLoading || !sessionReady}
        onInputChange={onInputChange}
        onKeyDown={onKeyDown}
        onSendMessage={onSendMessage}
        placeholder={
          pendingGate
            ? 'Ask about this step, or tell me what to change'
            : 'Ask any question or use voice input'
        }
      />
      </div>
      </div>

      {/* Structured gate panel (orchestrator mode only) */}
      <div
        className={`order-2 min-h-0 border-border bg-surface-sunken px-6 py-4 max-lg:border-t lg:flex lg:flex-col lg:overflow-y-auto ${
          pendingGate || shownSteps.length > 0 || isLoading ? '' : 'max-lg:hidden'
        }`}
      >
        <ReviewPanel doneSteps={doneSteps} files={files} isLoading={isLoading}>
          {pendingGate && onSubmitGateResolution ? (
            <div key={pendingGate.step} className="gate-in lg:flex lg:min-h-0 lg:flex-1 lg:flex-col">
              {pendingGate.step === 'confirm_eligibility' ? (
                <EligibilityGate
                  gate={pendingGate}
                  isLoading={isLoading}
                  onSubmit={onSubmitGateResolution}
                />
              ) : pendingGate.kind === 'choice' ? (
                <ChoiceGate
                  gate={pendingGate}
                  isLoading={isLoading}
                  onSubmit={onSubmitGateResolution}
                />
              ) : pendingGate.kind === 'input' ? (
                <InputGate
                  gate={pendingGate}
                  isLoading={isLoading}
                  onSubmit={onSubmitGateResolution}
                />
              ) : (
                <ApprovalGate
                  gate={pendingGate}
                  isLoading={isLoading}
                  onSubmit={onSubmitGateResolution}
                />
              )}
            </div>
          ) : null}
        </ReviewPanel>
      </div>
    </div>
  );
};

