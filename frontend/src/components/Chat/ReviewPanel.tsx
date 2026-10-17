/**
 * ReviewPanel - the right-hand pane of the split chat layout: tabs for the
 * open review step, the editor for the finished documents, a read-only view
 * of the CV, and the job summary.
 *
 * The Review tab shows the gate waiting for a decision; between gates it shows
 * the steps done so far and any files the run produced, so the pane is never
 * an empty box and the layout doesn't jump when a gate opens or closes.
 */

import { useState } from 'react';
import type { DownloadableFile, JobRequirements, ResumeInfo } from '@/types';
import { DocumentEditor, type RunDocumentSave } from '@/components/Editor/DocumentEditor';
import { DownloadButton } from './DownloadButton';
import { humanizeStep } from './gateFormat';

type Tab = 'review' | 'editor' | 'cv' | 'job';

interface ReviewPanelProps {
  /** Step of the open gate, or null between gates. */
  gateStep: string | null;
  /** The open gate's UI. */
  gate: React.ReactNode;
  /** True when the gate lays itself out and pins its own action bar (approval gates). */
  gateIsDocked: boolean;
  /** Gate steps the user has already been shown, oldest first. */
  doneSteps: string[];
  files: DownloadableFile[];
  isLoading: boolean;
  cvData?: ResumeInfo | null;
  jobData?: JobRequirements | null;
  /** The session, and how the editor saves through it; without them there is no editor. */
  sessionId?: string | null;
  runDocumentSave?: RunDocumentSave;
  /** Changes after every reply from the session, so the editor fetches the documents again. */
  refreshKey: number;
}

export const ReviewPanel = ({
  gateStep,
  gate,
  gateIsDocked,
  doneSteps,
  files,
  isLoading,
  cvData,
  jobData,
  sessionId,
  runDocumentSave,
  refreshKey,
}: ReviewPanelProps) => {
  const [tab, setTab] = useState<Tab>('review');
  // A new decision to make always brings the user back to the Review tab.
  // Adjusted while rendering (React's "state from props" pattern), not in an effect.
  const [lastStep, setLastStep] = useState<string | null>(null);
  if (gateStep !== lastStep) {
    setLastStep(gateStep);
    if (gateStep) setTab('review');
  }
  const canEdit = Boolean(sessionId && runDocumentSave) && files.some((f) => f.file_type === 'cover_letter');
  const tabs: { id: Tab; label: string }[] = [
    { id: 'review', label: 'Review' },
    ...(canEdit ? [{ id: 'editor' as const, label: 'Editor' }] : []),
    ...(cvData ? [{ id: 'cv' as const, label: 'CV preview' }] : []),
    ...(jobData ? [{ id: 'job' as const, label: 'Job summary' }] : []),
  ];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div role="tablist" aria-label="Review panel" className="flex shrink-0 gap-1 border-b border-border px-6">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            type="button"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px flex items-center gap-2 border-b-2 px-3 py-3 text-sm font-medium transition-colors ${
              tab === t.id ? 'border-primary text-ink' : 'border-transparent text-ink-muted hover:text-ink'
            }`}
          >
            {t.label}
            {t.id === 'review' && gateStep && (
              <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-primary px-1.5 text-meta font-semibold text-primary-fg">
                1
              </span>
            )}
          </button>
        ))}
      </div>

      {tab === 'review' &&
        (gateStep ? (
          <div
            key={gateStep}
            className={`gate-in flex min-h-0 flex-1 flex-col ${gateIsDocked ? '' : 'overflow-y-auto px-6 py-5'}`}
          >
            {gate}
          </div>
        ) : (
          <IdleReview
            doneSteps={doneSteps}
            files={files}
            isLoading={isLoading}
            onOpenEditor={canEdit ? () => setTab('editor') : undefined}
          />
        ))}
      {tab === 'editor' && canEdit && sessionId && runDocumentSave && (
        <DocumentEditor
          sessionId={sessionId}
          refreshKey={refreshKey}
          busy={isLoading}
          runDocumentSave={runDocumentSave}
        />
      )}
      {tab === 'cv' && cvData && <CvPreview cv={cvData} />}
      {tab === 'job' && jobData && <JobSummary job={jobData} />}
    </div>
  );
};

const IdleReview = ({
  doneSteps,
  files,
  isLoading,
  onOpenEditor,
}: Pick<ReviewPanelProps, 'doneSteps' | 'files' | 'isLoading'> & { onOpenEditor?: () => void }) => (
  <div className="flex-1 space-y-5 overflow-y-auto px-6 py-6">
    <div>
      <h2 className="text-lg font-semibold text-ink">{isLoading ? 'Working on it' : 'Nothing to review right now'}</h2>
      <p className="mt-1 text-sm text-ink-muted">
        {isLoading
          ? 'The next decision shows up here as soon as it is ready.'
          : doneSteps.length === 0
            ? 'Steps that need your decision show up here.'
            : onOpenEditor
              ? 'Edit your cover letter yourself, or ask in the chat to change something.'
              : 'Ask in the chat if you want to change something.'}
      </p>
      {onOpenEditor && !isLoading && (
        <button
          type="button"
          onClick={onOpenEditor}
          className="mt-3 rounded-full bg-primary px-4 py-2 text-sm font-semibold text-primary-fg transition-colors hover:bg-primary-hover"
        >
          Edit your cover letter
        </button>
      )}
    </div>
    {doneSteps.length > 0 && (
      <ol className="space-y-2" aria-label="Steps so far">
        {doneSteps.map((step) => (
          <li key={step} className="flex items-center gap-2.5 text-sm text-ink">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-success text-success-fg">
              <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </span>
            {humanizeStep(step)}
          </li>
        ))}
      </ol>
    )}
    {files.length > 0 && (
      <div>
        <p className="text-meta font-semibold uppercase tracking-[0.14em] text-ink-faint">Your files</p>
        {/* DownloadButton is inline-block, so without a gap the buttons sit on one line touching each other. */}
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {files.map((file) => (
            <DownloadButton key={file.download_url} file={file} />
          ))}
        </div>
      </div>
    )}
  </div>
);

const Heading = ({ children }: { children: React.ReactNode }) => (
  <h3 className="mb-2 mt-6 text-meta font-semibold uppercase tracking-[0.14em] text-ink-faint">{children}</h3>
);

const Pills = ({ items }: { items?: string[] }) =>
  items && items.length > 0 ? (
    <div className="flex flex-wrap gap-1.5">
      {items.map((s, i) => (
        <span key={`${s}-${i}`} className="rounded-full border border-border px-2.5 py-0.5 text-meta text-ink-muted">
          {s}
        </span>
      ))}
    </div>
  ) : null;

const Bullets = ({ items }: { items?: string[] }) =>
  items && items.length > 0 ? (
    <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-ink-muted">
      {items.map((b, i) => (
        <li key={i} className="[overflow-wrap:anywhere]">
          {b}
        </li>
      ))}
    </ul>
  ) : null;

const CvPreview = ({ cv }: { cv: ResumeInfo }) => (
  <div className="flex-1 overflow-y-auto px-6 py-6">
    <h2 className="text-lg font-semibold text-ink">{cv.name}</h2>
    <p className="text-sm text-ink-muted">{[cv.email, cv.phone, cv.location].filter(Boolean).join(' · ')}</p>
    {cv.summary && (
      <>
        <Heading>Summary</Heading>
        <p className="text-sm text-ink-muted">{cv.summary}</p>
      </>
    )}
    {cv.experience?.length > 0 && <Heading>Experience</Heading>}
    {cv.experience?.map((e, i) => (
      <div key={i} className="mb-3">
        <div className="text-sm font-medium text-ink">{[e.position, e.company].filter(Boolean).join(' · ')}</div>
        <div className="text-meta text-ink-faint">{[e.duration, e.location].filter(Boolean).join(' · ')}</div>
        <Bullets items={e.responsibilities} />
      </div>
    ))}
    {cv.projects && cv.projects.length > 0 && <Heading>Projects</Heading>}
    {cv.projects?.map((p, i) => (
      <div key={i} className="mb-3">
        <div className="text-sm font-medium text-ink">{p.name}</div>
        {p.description && <p className="text-sm text-ink-muted">{p.description}</p>}
        <Bullets items={p.outcomes} />
      </div>
    ))}
    {cv.education?.length > 0 && <Heading>Education</Heading>}
    {cv.education?.map((e, i) => (
      <div key={i} className="mb-3">
        <div className="text-sm font-medium text-ink">{[e.degree, e.institution].filter(Boolean).join(' · ')}</div>
        <div className="text-meta text-ink-faint">{[e.dates, e.grade].filter(Boolean).join(' · ')}</div>
      </div>
    ))}
    {cv.skills?.length > 0 && <Heading>Skills</Heading>}
    <Pills items={cv.skills} />
  </div>
);

const JobSummary = ({ job }: { job: JobRequirements }) => (
  <div className="flex-1 overflow-y-auto px-6 py-6">
    <h2 className="text-lg font-semibold text-ink">{job.job_title}</h2>
    <p className="text-sm text-ink-muted">
      {[job.job_level, job.employment_type, job.location, job.work_mode].filter(Boolean).join(' · ')}
    </p>
    {job.required_skills?.length > 0 && <Heading>Required skills</Heading>}
    <Pills items={job.required_skills} />
    {job.preferred_skills?.length > 0 && <Heading>Nice to have</Heading>}
    <Pills items={job.preferred_skills} />
    {job.responsibilities?.length > 0 && <Heading>Responsibilities</Heading>}
    <Bullets items={job.responsibilities} />
    {job.qualifications?.length > 0 && <Heading>Qualifications</Heading>}
    <Bullets items={job.qualifications} />
  </div>
);
