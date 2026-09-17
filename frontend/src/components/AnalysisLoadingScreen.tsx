'use client';

type StepState = 'pending' | 'active' | 'done' | 'skipped';

interface StepProps {
  title: string;
  description: string;
  state: StepState;
  showConnector?: boolean;
  nextStepSkipped?: boolean;
}

const Step = ({ title, description, state, showConnector = false, nextStepSkipped = false }: StepProps) => {
  const indicator = (() => {
    if (state === 'done') {
      return (
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-success text-success-fg">
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      );
    }

    if (state === 'skipped') {
      return (
        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border-strong bg-surface text-ink-muted">
          <span className="text-lg">–</span>
        </div>
      );
    }

    if (state === 'active') {
      return (
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-fg">
          <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        </div>
      );
    }

    return (
      <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border-strong bg-surface text-ink-faint">
        <span className="text-sm font-semibold">•</span>
      </div>
    );
  })();

  const titleClass =
    state === 'done'
      ? 'text-ink'
      : state === 'active'
        ? 'text-primary-text'
        : state === 'skipped'
          ? 'text-ink-muted'
          : 'text-ink-muted';

  const descClass = state === 'active' ? 'text-ink' : 'text-ink-muted';

  // Show connector if this step is done/skipped, OR if the next step is skipped (to show connection)
  const connectorVisible = state === 'done' || state === 'skipped' || nextStepSkipped;
  const connectorColor = connectorVisible ? 'bg-success' : 'bg-border-strong';

  return (
    <div className="relative">
      <div className="flex items-start gap-4">
        <div className="relative flex-shrink-0">
          {indicator}
          {/* Connecting Line - appears after THIS step is finished or when next step is skipped */}
          {showConnector && (
            <div
              className={`absolute left-1/2 top-10 h-6 w-0.5 -translate-x-1/2 origin-top transform transition-all duration-500 ${connectorColor} ${connectorVisible ? 'scale-y-100 opacity-100' : 'scale-y-0 opacity-0'}`}
            />
          )}
        </div>
        <div className="min-w-0 pt-0.5">
          <div className={`font-semibold ${titleClass}`}>{title}</div>
          <div className={`mt-0.5 text-label ${descClass}`}>{description}</div>
        </div>
      </div>
    </div>
  );
};

interface AnalysisLoadingScreenProps {
  resumeState: StepState;
  jobState: StepState;
  preparingState: StepState;
}

export const AnalysisLoadingScreen = ({
  resumeState,
  jobState,
  preparingState,
}: AnalysisLoadingScreenProps) => {
  const headline =
    resumeState === 'active'
      ? 'Extracting your resume'
      : jobState === 'active'
        ? 'Analyzing the job description'
      : preparingState === 'active'
        ? 'Preparing your AI assistant'
        : 'Starting analysis';

  const subheadline =
    resumeState === 'active'
      ? 'We’re extracting skills, experience, and achievements.'
      : jobState === 'active'
        ? 'Extracting key requirements, skills, and keywords to tailor your documents.'
      : preparingState === 'active'
        ? 'Almost ready — setting up your chat workspace.'
        : 'This usually takes a few moments.';

  return (
    <div className="flex flex-1 items-center justify-center py-10">
      <div className="w-full max-w-3xl">
        <div className="rounded-2xl border border-border bg-surface p-10 shadow-lg shadow-black/5">
          <div className="mx-auto max-w-xl text-center">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary-surface text-primary-text">
              <svg className="h-9 w-9 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            </div>
            <h2 className="text-2xl font-semibold tracking-title text-ink">
              {headline}
            </h2>
            <p className="mx-auto mt-2 max-w-[42ch] text-body text-ink-muted">{subheadline}</p>
          </div>

          <div className="mt-10 space-y-6">
            <Step
              title="Resume extraction"
              description="Parsing your PDF and structuring your profile."
              state={resumeState}
              showConnector={true}
              nextStepSkipped={jobState === 'skipped'}
            />
            {jobState !== 'skipped' && (
              <Step
                title="Job description"
                description={
                  jobState === 'active'
                    ? 'Extracting requirements and keywords to match your resume.'
                    : 'Using job requirements to tailor your documents (optional).'
                }
                state={jobState}
                showConnector={true}
              />
            )}
            <Step
              title="Prepare chat"
              description={
                preparingState === 'active'
                  ? 'Initializing your AI assistant and generating your resume summary.'
                  : 'Loading your personalized assistant and tools.'
              }
              state={preparingState}
              showConnector={false}
            />
          </div>

          <div className="mt-10 rounded-xl border border-border bg-surface-sunken p-4 text-sm text-ink-muted">
            You can keep this tab open — we’ll continue automatically.
          </div>
        </div>
      </div>
    </div>
  );
};
