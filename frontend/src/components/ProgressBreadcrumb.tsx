'use client';

interface ProgressBreadcrumbProps {
  resumeUploaded: boolean;
  jobUploaded: boolean;
  jobSkipped: boolean;
  jobInputValid: boolean;
  analysisStarted: boolean;
  isAnalyzing: boolean;
  chatReady: boolean;
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
      label: 'Chat Ready',
      completed: chatReady,
      current: analysisStarted && !isAnalyzing && !chatReady,
    },
  ];

  const stages = allStages;

  return (
    <div className="w-full overflow-x-auto">
      <div className="flex min-w-max items-center justify-center px-2">
        {stages.map((stage, index) => (
          <div key={stage.id} className="flex shrink-0 items-center">
            {/* Stage Circle */}
            <div className="flex items-center gap-3">
              <div
                className={`flex h-10 w-10 items-center justify-center rounded-full border-2 transition-colors ${
                  stage.completed
                    ? 'border-success bg-success'
                    : stage.current
                      ? 'border-primary bg-primary'
                      : 'border-border-strong bg-surface'
                }`}
              >
                {stage.completed ? (
                  <svg
                    className="h-5 w-5 text-success-fg"
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
                  <div className="h-3 w-3 rounded-full bg-primary-fg"></div>
                ) : (
                  <span className="text-sm font-medium tabular-nums text-ink-faint">
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
                className={`mx-2 h-0.5 w-8 transition-colors sm:mx-4 sm:w-16 ${
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
