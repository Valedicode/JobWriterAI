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
    <div className="flex flex-1 flex-col rounded-xl border border-border bg-surface shadow-sm">
      <div className="flex flex-1 flex-col items-center justify-start pt-16 sm:pt-20 md:pt-24 lg:pt-32">
        <div className="mx-auto w-full max-w-3xl px-6">
          {/* Question */}
          <div className="mb-10 text-center">
            <h2 className="mx-auto mb-3 max-w-[18ch] text-balance text-4xl font-bold tracking-display text-ink sm:text-5xl">
              What would you like to work on today?
            </h2>
            <p className="text-lede text-ink-muted">
              Upload your resume and ask me anything to get started
            </p>
          </div>

          {/* Large Input Field */}
          <div className="w-full">
            <div className="relative flex items-center rounded-2xl border border-border-strong bg-surface px-4 py-4 shadow-sm transition-colors focus-within:border-primary focus-within:shadow-md">
              {/* Plus Icon */}
              <button
                onClick={onClickUpload}
                className="mr-3 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg text-ink-faint transition-colors hover:bg-surface-sunken hover:text-ink"
                aria-label="Add attachment"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>
              
              {/* Text Input */}
              <textarea
                ref={textareaRef}
                value={inputText}
                onChange={(e) => onInputChange(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask any question"
                maxLength={4000}
                className="flex-1 resize-none border-0 bg-transparent text-base text-ink placeholder:text-ink-faint focus:outline-none focus:ring-0"
                rows={1}
                style={{ minHeight: '32px', maxHeight: '200px' }}
              />

              {/* Right Side Icons */}
              <div className="ml-3 flex items-center gap-2">
                {/* Send Button */}
                <button
                  onClick={onSendMessage}
                  disabled={!inputText.trim() || isLoading}
                  className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-fg transition-transform hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50 ${
                    inputText.trim() && !isLoading ? 'hover:scale-105' : ''
                  }`}
                  aria-label="Send message"
                >
                  {isLoading ? (
                    <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>
          
          {/* Quick Suggestions */}
          <QuickSuggestions onSuggestionClick={onInputChange} />
        </div>
      </div>
    </div>
  );
};

