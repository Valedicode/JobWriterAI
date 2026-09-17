import { Message, OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { EmptyState } from './EmptyState';
import { ApprovalGate } from './ApprovalGate';
import { ChoiceGate } from './ChoiceGate';
import { InputGate } from './InputGate';
import { ErrorBoundary } from '@/components/ErrorBoundary';

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
    opts?: { feedback?: string; choice?: string }
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
    <div className="flex flex-1 flex-col rounded-2xl border border-border bg-surface shadow-sm animate-fade-in">
      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-6">
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

      {/* Structured gate panel (orchestrator mode only) */}
      {pendingGate && onSubmitGateResolution && (
        <div
          key={pendingGate.step}
          className="gate-in border-t border-border bg-surface-sunken px-6 py-4"
        >
          {pendingGate.kind === 'choice' ? (
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
      )}

      {/* Input Area */}
      <ChatInput
        textareaRef={textareaRef}
        inputText={inputText}
        isLoading={isLoading || !sessionReady}
        onInputChange={onInputChange}
        onKeyDown={onKeyDown}
        onSendMessage={onSendMessage}
      />
    </div>
  );
};

