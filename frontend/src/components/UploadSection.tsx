'use client';

import { ResumeUpload } from '@/components/ResumeUpload';
import { JobInput } from '@/components/JobInput';
import { HowItWorks } from '@/components/HowItWorks';
import { LandingHero } from '@/components/landing/LandingHero';
import { FlowChooser } from '@/components/landing/FlowChooser';
import type { ResumeInfo, JobRequirements } from '@/types';

interface UploadSectionProps {
  // Flow mode selection
  flowMode: 'cv_only' | 'job_tailoring' | null;
  onSetFlowMode: (mode: 'cv_only' | 'job_tailoring') => void;

  // Resume upload props
  uploadedFile: File | null;
  isDragging: boolean;
  uploadError: string | null;
  isUploading: boolean;
  cvData: ResumeInfo | null;
  needsClarification: boolean;
  clarificationQuestions: string[] | null;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClickUpload: () => void;
  onRemoveFile: () => void;

  // Job input props
  jobUrl: string;
  setJobUrl: (url: string) => void;
  jobText: string;
  setJobText: (text: string) => void;
  jobData: JobRequirements | null;
  isJobProcessing: boolean;
  jobError: string | null;
  urlValidationError: string | null;
  textValidationError: string | null;
  onJobClear: () => void;

  // Manual start props
  onSkipJob: () => void;
  onStartAnalysis: () => void;
  analysisStarted: boolean;
  canStartAnalysis: boolean;
}

export const UploadSection = ({
  flowMode,
  onSetFlowMode,
  uploadedFile,
  isDragging,
  uploadError,
  isUploading,
  cvData,
  needsClarification,
  clarificationQuestions,
  fileInputRef,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileInputChange,
  onClickUpload,
  onRemoveFile,
  jobUrl,
  setJobUrl,
  jobText,
  setJobText,
  jobData,
  isJobProcessing,
  jobError,
  urlValidationError,
  textValidationError,
  onJobClear,
  onSkipJob,
  onStartAnalysis,
  analysisStarted,
  canStartAnalysis,
}: UploadSectionProps) => {
  // Landing: split showcase hero + the three named flows.
  if (flowMode === null) {
    return (
      <div className="flex flex-1 flex-col items-center">
        <LandingHero onStart={() => onSetFlowMode('job_tailoring')} />
        <FlowChooser onSetFlowMode={onSetFlowMode} />
        <div className="w-full max-w-5xl px-6 pb-16">
          <HowItWorks />
        </div>
      </div>
    );
  }

  // A flow is chosen — upload the resume (and, for tailoring, the job posting).
  return (
    <div className="flex flex-1 flex-col items-center">
      <div className="w-full max-w-4xl px-6 py-12 text-center">
        <h1 className="mx-auto mb-5 max-w-[18ch] text-balance font-serif text-4xl font-normal leading-[1.08] tracking-[-0.005em] text-ink sm:text-5xl">
          {flowMode === 'cv_only'
            ? 'Review your CV, step by step'
            : 'Tailor your application to the job'}
        </h1>
        <p className="mx-auto mb-12 max-w-[48ch] text-balance text-lede text-ink-muted">
          {flowMode === 'cv_only'
            ? 'Upload your resume for a structured quality review covering clarity, ATS readiness, quantified impact, and formatting.'
            : 'Upload your resume and the job posting to generate a tailored resume and a matching cover letter — reviewed at every gate.'}
        </p>

        <div className="mx-auto w-full max-w-4xl space-y-8">
          <div
            className={
              flowMode === 'cv_only'
                ? 'flex justify-center'
                : 'grid gap-8 lg:grid-cols-2'
            }
          >
            {/* Resume Upload Card */}
            <div
              className={`rounded-2xl border border-border bg-surface p-8 shadow-lg shadow-black/5${
                flowMode === 'cv_only' ? ' w-full max-w-lg' : ''
              }`}
            >
              <div className="mb-6 flex items-center justify-center">
                <div className="rounded-full bg-primary-surface p-6">
                  <svg
                    className="h-14 w-14 text-primary-text"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
              </div>
              <h2 className="mb-3 font-serif text-2xl font-normal text-ink">
                Your resume
              </h2>
              <p className="mb-6 text-body text-ink-muted">
                Upload your current resume as a PDF file.
              </p>
              <ResumeUpload
                uploadedFile={uploadedFile}
                isDragging={isDragging}
                uploadError={uploadError}
                isUploading={isUploading}
                cvData={cvData}
                needsClarification={needsClarification}
                clarificationQuestions={clarificationQuestions}
                fileInputRef={fileInputRef}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                onFileInputChange={onFileInputChange}
                onClickUpload={onClickUpload}
                onRemoveFile={onRemoveFile}
              />
            </div>

            {/* Job Description Card — only shown in job_tailoring flow */}
            {flowMode === 'job_tailoring' && (
              <div className="rounded-2xl border border-border bg-surface p-8 shadow-lg shadow-black/5">
                <div className="mb-6 flex items-center justify-center">
                  <div className="rounded-full bg-primary-surface p-6">
                    <svg
                      className="h-14 w-14 text-primary-text"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                </div>
                <div className="mb-3 flex items-center justify-center gap-2">
                  <h2 className="font-serif text-2xl font-normal text-ink">
                    Job description
                  </h2>
                </div>
                <p className="mb-6 text-body text-ink-muted">
                  Paste the job description or provide a URL.
                </p>

                <JobInput
                  jobUrl={jobUrl}
                  setJobUrl={setJobUrl}
                  jobText={jobText}
                  setJobText={setJobText}
                  jobData={jobData}
                  isProcessing={isJobProcessing}
                  error={jobError}
                  urlValidationError={urlValidationError}
                  textValidationError={textValidationError}
                  onClear={onJobClear}
                />

                {!jobData && !isJobProcessing && (
                  <button
                    type="button"
                    onClick={onSkipJob}
                    className="mt-4 w-full rounded-lg border border-border-strong bg-surface px-4 py-2.5 text-sm font-medium text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink"
                  >
                    Don&apos;t have a posting? Review your CV instead
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Start Analysis Button */}
          {!analysisStarted && uploadedFile && (
            <div className="flex flex-col items-center gap-4">
              <button
                onClick={onStartAnalysis}
                disabled={!canStartAnalysis || isUploading || isJobProcessing}
                className="rounded-full bg-primary px-8 py-4 text-base font-semibold text-primary-fg shadow-[0_14px_32px_-12px] shadow-primary/50 transition-transform duration-200 hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
              >
                {isUploading || isJobProcessing ? (
                  <span className="flex items-center gap-3">
                    <svg
                      className="h-5 w-5 animate-spin"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Processing…
                  </span>
                ) : (
                  <span className="flex items-center gap-3">
                    <svg
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                    {flowMode === 'cv_only' ? 'Start CV review' : 'Start analysis'}
                  </span>
                )}
              </button>

              <p className="text-center text-body text-ink-muted">
                {flowMode === 'cv_only'
                  ? 'Review your CV for quality, clarity, and ATS readiness.'
                  : 'Analyze your resume against the job requirements.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
