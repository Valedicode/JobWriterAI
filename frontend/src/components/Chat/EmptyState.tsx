import { QuickSuggestions } from './QuickSuggestions';

interface EmptyStateProps {
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  inputText: string;
  isLoading: boolean;
  onInputChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSendMessage: () => void;
  onClickUpload: () => void;
}

// Outer node owns position + tilt; inner node owns the float so the
// translateY keyframe never fights the rotation.
const FloatingBubble = ({
  children,
  side,
}: {
  children: React.ReactNode;
  side: 'left' | 'right';
}) => (
  <div
    aria-hidden="true"
    className={`absolute hidden max-w-[15rem] lg:block ${
      side === 'left' ? 'left-0 top-4 -rotate-2' : 'right-0 top-14 rotate-2'
    }`}
  >
    <div
      style={{ animationDelay: side === 'left' ? '0s' : '1.4s' }}
      className={`float-idle rounded-2xl border border-border bg-surface px-3.5 py-2 text-sm text-ink-muted shadow-[0_16px_36px_-18px_rgba(60,38,22,0.28)] ${
        side === 'left' ? 'rounded-bl-sm' : 'rounded-br-sm'
      }`}
    >
      {children}
    </div>
  </div>
);

export const EmptyState = ({
  textareaRef,
  inputText,
  isLoading,
  onInputChange,
  onKeyDown,
  onSendMessage,
  onClickUpload,
}: EmptyStateProps) => {
  return (
    <div className="flex flex-1 flex-col rounded-2xl border border-border bg-surface shadow-sm animate-fade-in">
      <div className="flex flex-1 flex-col items-center justify-center px-6 py-12 sm:py-16">
        <div className="mx-auto w-full max-w-2xl">
          {/* Greeting + the calm illustrative moment */}
          <div className="relative mb-10 text-center">
            <div className="relative mx-auto mb-8 hidden h-28 max-w-lg lg:block">
              <FloatingBubble side="left">
                Make my experience section punchier.
              </FloatingBubble>
              <div className="absolute left-1/2 top-2 -ml-8 h-16 w-16">
               <div className="float-idle-slow flex h-16 w-16 items-center justify-center rounded-full bg-primary-surface">
                <svg
                  className="h-7 w-7 text-primary-text"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    d="M12 3l2.5 5.5L20 11l-5.5 2.5L12 19l-2.5-5.5L4 11l5.5-2.5L12 3z"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
               </div>
              </div>
              <FloatingBubble side="right">
                Sure — and you&rsquo;ll approve every change first.
              </FloatingBubble>
            </div>

            <h2 className="mx-auto max-w-[20ch] text-balance font-serif text-3xl font-normal leading-[1.1] tracking-[-0.005em] text-ink sm:text-4xl">
              Let&rsquo;s get your resume in shape.
            </h2>
            <p className="mx-auto mt-3 max-w-[44ch] text-lede text-ink-muted">
              Ask a question or paste a job posting. Nothing is written to your
              resume until you approve it.
            </p>
          </div>

          {/* Composer */}
          <div className="rounded-2xl border border-border-strong bg-surface p-3 shadow-sm transition-colors focus-within:border-primary focus-within:shadow-md">
            <textarea
              ref={textareaRef}
              value={inputText}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about your resume, or paste a job posting…"
              maxLength={4000}
              className="w-full resize-none border-0 bg-transparent px-2 py-1.5 text-base text-ink placeholder:text-ink-faint focus:outline-none focus:ring-0"
              rows={2}
              style={{ minHeight: '52px', maxHeight: '200px' }}
            />

            <div className="mt-1 flex items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={onClickUpload}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-sm text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                    <path d="M12 16V4m0 0L8 8m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Attach resume
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onInputChange('Here is the job posting I am targeting:\n\n');
                    textareaRef.current?.focus();
                  }}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-sm text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                    <path d="M9 5h6a2 2 0 012 2v12a2 2 0 01-2 2H9a2 2 0 01-2-2V7a2 2 0 012-2zm0 0V4a1 1 0 011-1h4a1 1 0 011 1v1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Paste job posting
                </button>
              </div>

              <button
                type="button"
                onClick={onSendMessage}
                disabled={!inputText.trim() || isLoading}
                className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-fg transition-transform duration-200 hover:-translate-y-0.5 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
                aria-label="Send message"
              >
                {isLoading ? (
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                ) : (
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                    <path d="M5 12h13M12 5l7 7-7 7" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <QuickSuggestions onSuggestionClick={onInputChange} />
        </div>
      </div>
    </div>
  );
};
