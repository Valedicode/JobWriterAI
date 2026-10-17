import { useState } from 'react';
import {
  BulletDecision,
  JobRequirements,
  Message,
  OrchestratorGateAction,
  OrchestratorGatePayload,
  ResumeInfo,
} from '@/types';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { EmptyState } from './EmptyState';
import { ApprovalGate } from './ApprovalGate';
import { ChoiceGate } from './ChoiceGate';
import { InputGate } from './InputGate';
import { EligibilityGate } from './EligibilityGate';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { ReviewPanel } from './ReviewPanel';
import { humanizeStep } from './gateFormat';

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
  /** Shown read-only in the review pane's CV preview / Job summary tabs. */
  cvData?: ResumeInfo | null;
  jobData?: JobRequirements | null;
  // Orchestrator-only: when set, render the structured gate in the review pane
  // next to the chat. Legacy writer chat leaves these unset and the UI behaves
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
  cvData,
  jobData,
  pendingGate,
  onSubmitGateResolution,
}: ChatContainerProps) => {
  // Gate steps shown so far, for the review pane's step list. Updated while
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

  const gate = pendingGate && onSubmitGateResolution ? pendingGate : null;
  const gateIsDocked =
    !!gate && gate.kind !== 'choice' && gate.kind !== 'input' && gate.step !== 'confirm_eligibility';

  return (
    // Below lg: one column (messages, step, input). From lg: chat on the left,
    // review pane on the right, filling the screen and scrolling separately.
    <div className="flex min-h-0 flex-1 flex-col animate-fade-in lg:grid lg:grid-cols-[minmax(0,5fr)_minmax(0,9fr)]">
      <div className="contents lg:flex lg:min-h-0 lg:flex-col lg:border-r lg:border-border">
        {/* Chat Messages Area */}
        <div className="order-1 min-h-0 flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-3xl">
            <div className="space-y-4" role="log" aria-live="polite" aria-label="Conversation">
              {messages.map((message) => (
                <ErrorBoundary
                  key={message.id}
                  fallback={<p className="text-sm text-ink-muted">(A message couldn’t be displayed.)</p>}
                >
                  <ChatMessage
                    message={message}
                    generatedFiles={message.generatedFiles}
                    isFadingOut={fadingOutMessageId === message.id}
                  />
                </ErrorBoundary>
              ))}

              {/* Points at the review pane; below lg the step sits right under the chat. */}
              {gate && (
                <div className="hidden rounded-xl border border-dashed border-primary/60 px-4 py-3 text-sm text-ink lg:block">
                  {humanizeStep(gate.step)} is ready.
                  <span className="mt-1 block text-meta font-medium text-primary-text">→ Review on the right</span>
                </div>
              )}

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
        <div className="order-3 shrink-0">
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

      {/* Review pane: the step waiting for a decision, CV preview, job summary */}
      <section className="order-2 flex min-h-0 flex-col bg-surface-sunken max-lg:max-h-[60%] max-lg:border-t max-lg:border-border">
        <ReviewPanel
          gateStep={gate ? gate.step : null}
          gateIsDocked={gateIsDocked}
          doneSteps={doneSteps}
          files={files}
          isLoading={isLoading}
          cvData={cvData}
          jobData={jobData}
          gate={
            gate && onSubmitGateResolution ? (
              gate.step === 'confirm_eligibility' ? (
                <EligibilityGate gate={gate} isLoading={isLoading} onSubmit={onSubmitGateResolution} />
              ) : gate.kind === 'choice' ? (
                <ChoiceGate gate={gate} isLoading={isLoading} onSubmit={onSubmitGateResolution} />
              ) : gate.kind === 'input' ? (
                <InputGate gate={gate} isLoading={isLoading} onSubmit={onSubmitGateResolution} />
              ) : (
                <ApprovalGate gate={gate} isLoading={isLoading} onSubmit={onSubmitGateResolution} />
              )
            ) : null
          }
        />
      </section>
    </div>
  );
};
