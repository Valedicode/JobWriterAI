import type { JobRequirements } from '@/types';

interface JobInputProps {
  jobUrl: string;
  setJobUrl: (url: string) => void;
  jobText: string;
  setJobText: (text: string) => void;
  jobData: JobRequirements | null;
  isProcessing: boolean;
  error: string | null;
  urlValidationError: string | null;
  textValidationError: string | null;
  onClear: () => void;
}

export const JobInput = ({
  jobUrl,
  setJobUrl,
  jobText,
  setJobText,
  jobData,
  isProcessing,
  error,
  urlValidationError,
  textValidationError,
  onClear,
}: JobInputProps) => {
  return (
    <div>
      {/* Input Area */}
      {!jobData ? (
        <div className="space-y-4">
          {/* URL Input */}
          <div className="relative">
            <label htmlFor="job-url" className="mb-2 block text-sm font-medium text-ink">
              Job Posting URL (Optional)
            </label>
            <input
              id="job-url"
              type="url"
              value={jobUrl}
              onChange={(e) => setJobUrl(e.target.value)}
              placeholder="https://example.com/job-posting"
              aria-invalid={!!urlValidationError}
              className={`w-full rounded-xl border bg-surface-sunken px-4 py-3 text-[16px] text-ink placeholder:text-ink-faint focus:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 sm:text-sm ${
                urlValidationError
                  ? 'border-danger focus:border-danger'
                  : 'border-border-strong focus:border-primary'
              }`}
            />
            {urlValidationError && (
              <p className="mt-1.5 text-meta text-danger-text">
                {urlValidationError}
              </p>
            )}
            {!urlValidationError && jobUrl.trim() && (
              <p className="mt-1.5 text-meta text-success-text">
                Valid URL
              </p>
            )}
          </div>

          {/* Text Input */}
          <div className="relative">
            <label htmlFor="job-text" className="mb-2 block text-sm font-medium text-ink">
              Job Description Text (Optional)
            </label>
            <textarea
              id="job-text"
              value={jobText}
              onChange={(e) => setJobText(e.target.value)}
              placeholder="Paste the job description text here (minimum 50 characters)..."
              aria-invalid={!!textValidationError}
              maxLength={20000}
              className={`min-h-[200px] w-full resize-none rounded-xl border bg-surface-sunken px-4 py-4 text-[16px] text-ink placeholder:text-ink-faint focus:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-ring/30 sm:text-sm ${
                textValidationError
                  ? 'border-danger focus:border-danger'
                  : 'border-border-strong focus:border-primary'
              }`}
            />
            <div className="mt-2 flex items-center justify-between">
              <p className="text-meta tabular-nums text-ink-muted">
                {textValidationError ? (
                  <span className="text-danger-text">{textValidationError}</span>
                ) : jobText.trim().length > 0 ? (
                  <span className={jobText.trim().length >= 50 ? 'text-success-text' : 'text-ink-muted'}>
                    {jobText.trim().length} characters
                  </span>
                ) : (
                  'Enter at least 50 characters'
                )}
              </p>
              {jobText.trim().length >= 50 && (
                <p className="text-meta text-success-text">Valid</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Success Message */}
          <div role="status" className="rounded-lg border border-success-border bg-success-surface p-4">
            <div className="flex items-start gap-3">
              <svg
                className="h-5 w-5 flex-shrink-0 text-success-text"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-success-text">
                  Job analyzed successfully!
                </p>
                <p className="mt-0.5 text-sm text-success-text [overflow-wrap:anywhere]">
                  {[jobData.job_title, jobData.location].filter(Boolean).join(' — ') || 'Details below'}
                </p>
              </div>
            </div>
          </div>

          {/* Job Details Preview */}
          <div className="rounded-lg border border-border bg-surface-sunken p-4">
            <div className="space-y-2 text-sm">
              <div className="[overflow-wrap:anywhere]">
                <span className="font-medium text-ink">
                  Title:
                </span>{' '}
                <span className="text-ink-muted">
                  {jobData.job_title || '—'}
                </span>
              </div>
              <div className="[overflow-wrap:anywhere]">
                <span className="font-medium text-ink">
                  Level:
                </span>{' '}
                <span className="text-ink-muted">
                  {jobData.job_level || '—'}
                </span>
              </div>
              <div className="[overflow-wrap:anywhere]">
                <span className="font-medium text-ink">
                  Required Skills:
                </span>{' '}
                <span className="text-ink-muted">
                  {(jobData.required_skills ?? []).length > 0
                    ? (jobData.required_skills ?? []).slice(0, 3).join(', ') +
                      ((jobData.required_skills ?? []).length > 3 ? '…' : '')
                    : '—'}
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={onClear}
            className="w-full rounded-lg border border-border-strong bg-surface px-4 py-2.5 text-sm font-medium text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink"
          >
            Clear & Enter New Job
          </button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div role="alert" className="mt-4 rounded-lg border border-danger-border bg-danger-surface p-4">
          <div className="flex items-start gap-3">
            <svg
              className="h-5 w-5 flex-shrink-0 text-danger-text"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm font-medium text-danger-text">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};



