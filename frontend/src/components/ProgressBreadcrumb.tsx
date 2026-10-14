'use client';

interface ProgressBreadcrumbProps {
  resumeUploaded: boolean;
  jobUploaded: boolean;
  jobSkipped: boolean;
  jobInputValid: boolean;
  analysisStarted: boolean;
  isAnalyzing: boolean;
  chatReady: boolean;
  /** Slim variant for the chat screen's top bar. */
  compact?: boolean;
}

interface Stage {
  id: string;
  label: string;
  completed: boolean;
  current: boolean;
}

export const ProgressBreadcrumb = ({
  resumeUploaded,
  jobUploaded,
  jobSkipped,
  jobInputValid,
  analysisStarted,
  isAnalyzing,
  chatReady,
  compact = false,
}: ProgressBreadcrumbProps) => {
  // Build stages dynamically - exclude job step if skipped
  const allStages: Stage[] = [
    {
      id: 'resume',
      label: 'Upload Resume',
      completed: resumeUploaded,
      current: !resumeUploaded,
    },
    // Only include job stage if not skipped
    ...(!jobSkipped ? [{
      id: 'job',
      label: 'Add Job Description',
      completed: jobUploaded || jobInputValid,
      current: resumeUploaded && !jobUploaded && !jobInputValid && !analysisStarted,
    }] : []),
    {
      id: 'analyze',
      label: 'Analysis',
      completed: analysisStarted && !isAnalyzing,
      current: isAnalyzing,
    },
    {
      id: 'chat',
      label: compact ? 'Review & tailor' : 'Chat Ready',
      // In the top bar the last step is where the user works, so it stays current.
      completed: compact ? false : chatReady,
      current: compact ? chatReady : analysisStarted && !isAnalyzing && !chatReady,
    },
  ];

  const stages = allStages;

  return (
    <div className="w-full overflow-x-auto">
      <div className={`flex min-w-max items-center px-2 ${compact ? 'justify-start' : 'justify-center'}`}>
        {stages.map((stage, index) => (
          <div key={stage.id} className="flex shrink-0 items-center">
            {/* Stage Circle */}
            <div className={`flex items-center ${compact ? 'gap-2' : 'gap-3'}`}>
              <div
                className={`flex items-center ${compact ? 'h-6 w-6' : 'h-10 w-10'} justify-center rounded-full border-2 transition-colors ${
                  stage.completed
                    ? 'border-success bg-success'
                    : stage.current
                      ? 'border-primary bg-primary'
                      : 'border-border-strong bg-surface'
                }`}
              >
                {stage.completed ? (
                  <svg
                    className={compact ? 'h-3.5 w-3.5 text-success-fg' : 'h-5 w-5 text-success-fg'}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2.5}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                ) : stage.current ? (
                  <div className={`rounded-full bg-primary-fg ${compact ? 'h-2 w-2' : 'h-3 w-3'}`}></div>
                ) : (
                  <span className={`font-medium tabular-nums text-ink-faint ${compact ? 'text-meta' : 'text-sm'}`}>
                    {index + 1}
                  </span>
                )}
              </div>

              {/* Stage Label */}
              <span
                className={`hidden text-sm font-medium transition-colors sm:block ${
                  stage.completed
                    ? 'text-success-text'
                    : stage.current
                      ? 'text-primary-text'
                      : 'text-ink-faint'
                }`}
              >
                {stage.label}
              </span>
            </div>

            {/* Connector Line */}
            {index < stages.length - 1 && (
              <div
                className={`h-0.5 transition-colors ${
                  compact ? 'mx-2 w-6' : 'mx-2 w-8 sm:mx-4 sm:w-16'
                } ${
                  stage.completed ? 'bg-success' : 'bg-border-strong'
                }`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
