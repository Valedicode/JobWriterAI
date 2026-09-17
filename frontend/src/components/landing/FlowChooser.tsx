'use client';

import { useState } from 'react';

interface FlowChooserProps {
  onSetFlowMode: (mode: 'cv_only' | 'job_tailoring') => void;
}

const ReviewIcon = () => (
  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M9 12h6m-6 4h4m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const TailorIcon = () => (
  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2zM12 11v6M9 14h6"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const DiscoverIcon = () => (
  <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14zM13.5 8.5l-1.2 3.3-3.3 1.2 1.2-3.3 3.3-1.2z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const Arrow = () => (
  <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
    <path
      d="M4 10h11M11 5l5 5-5 5"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export const FlowChooser = ({ onSetFlowMode }: FlowChooserProps) => {
  const [discoveryNote, setDiscoveryNote] = useState(false);

  return (
    <section className="w-full px-6 pb-20">
      <div className="mx-auto max-w-5xl rounded-[24px] border border-border bg-surface-sunken px-6 py-11 sm:px-10">
        <h2 className="mb-8 text-center font-serif text-3xl font-normal text-ink">
          Or choose where to start
        </h2>

        <div className="relative grid rounded-[18px] border border-border bg-surface sm:grid-cols-3 sm:divide-x sm:divide-border">
          {/* Review my CV */}
          <button
            type="button"
            onClick={() => onSetFlowMode('cv_only')}
            className="group flex flex-col items-start p-7 text-left transition-colors hover:bg-primary-surface/30"
          >
            <span className="mb-4 inline-flex h-9 w-9 items-center justify-center rounded-full bg-primary-surface text-primary-text">
              <ReviewIcon />
            </span>
            <h3 className="font-serif text-lg font-normal text-ink">
              Review my CV
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
              A structured quality review — clarity, ATS readiness, quantified
              impact, formatting.
            </p>
            <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-primary-text">
              Start a review
              <span className="transition-transform group-hover:translate-x-0.5">
                <Arrow />
              </span>
            </span>
          </button>

          {/* Tailor to a job — featured */}
          <button
            type="button"
            onClick={() => onSetFlowMode('job_tailoring')}
            className="group relative z-10 flex flex-col items-start bg-surface p-7 text-left transition-colors hover:bg-primary-surface/30 sm:-my-3 sm:rounded-[18px] sm:border-2 sm:border-primary sm:shadow-[0_18px_44px_-18px_rgba(60,38,22,0.3)]"
          >
            <span className="mb-4 inline-flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-fg">
              <TailorIcon />
            </span>
            <h3 className="font-serif text-lg font-normal text-ink">
              Tailor to a job
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
              Match your resume to one specific posting and generate a tailored
              CV and cover letter.
            </p>
            <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-primary-text">
              Start tailoring
              <span className="transition-transform group-hover:translate-x-0.5">
                <Arrow />
              </span>
            </span>
          </button>

          {/* Discover careers — not yet available */}
          <button
            type="button"
            onClick={() => setDiscoveryNote(true)}
            aria-disabled="true"
            className="group relative flex cursor-not-allowed flex-col items-start p-7 text-left"
          >
            <span className="mb-4 inline-flex h-9 w-9 items-center justify-center rounded-full border border-dashed border-border-strong text-ink-faint">
              <DiscoverIcon />
            </span>
            <h3 className="font-serif text-lg font-normal text-ink-muted">
              Discover careers
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-muted">
              Explore the roles and trajectories that fit the skills you already
              have.
            </p>
            <span className="mt-4 text-sm font-medium text-ink-muted">
              Coming soon
            </span>
          </button>
        </div>

        {discoveryNote && (
          <p
            role="status"
            className="mx-auto mt-6 max-w-xl text-center text-sm text-ink-muted"
          >
            Career discovery isn&rsquo;t available yet. For now, start with{' '}
            <button
              type="button"
              className="font-medium text-primary-text underline underline-offset-2"
              onClick={() => onSetFlowMode('cv_only')}
            >
              a CV review
            </button>{' '}
            or{' '}
            <button
              type="button"
              className="font-medium text-primary-text underline underline-offset-2"
              onClick={() => onSetFlowMode('job_tailoring')}
            >
              tailoring to a job
            </button>
            .
          </p>
        )}
      </div>
    </section>
  );
};
