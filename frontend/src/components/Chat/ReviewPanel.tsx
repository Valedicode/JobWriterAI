/**
 * ReviewPanel - the right-hand side of the split chat layout. Shows the gate
 * waiting for the user's decision; between gates it shows the steps done so
 * far and any files the run produced, so the panel is never an empty box and
 * the layout doesn't jump when a gate opens or closes.
 */

import type { DownloadableFile } from '@/types';
import { DownloadButton } from './DownloadButton';
import { humanizeStep } from './gateFormat';

interface ReviewPanelProps {
  /** Gate steps the user has already been shown, oldest first. */
  doneSteps: string[];
  files: DownloadableFile[];
  isLoading: boolean;
  /** The open gate, or null between gates. */
  children?: React.ReactNode;
}

export const ReviewPanel = ({ doneSteps, files, isLoading, children }: ReviewPanelProps) => {
  if (children) return <>{children}</>;

  return (
    <div className="flex h-full flex-col gap-5 overflow-y-auto">
      <div>
        <p className="text-meta font-semibold uppercase tracking-[0.14em] text-primary-text">
          {isLoading ? 'Working on it' : 'Your review'}
        </p>
        <p className="mt-1 text-sm text-ink-muted">
          {isLoading
            ? 'The next decision shows up here as soon as it is ready.'
            : doneSteps.length === 0
              ? 'Steps that need your decision show up here.'
              : 'Nothing needs your decision right now. Ask in the chat if you want to change something.'}
        </p>
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
        <div className="space-y-2">
          <p className="text-meta font-semibold uppercase tracking-[0.14em] text-ink-faint">Your files</p>
          {files.map((file) => (
            <DownloadButton key={file.download_url} file={file} />
          ))}
        </div>
      )}
    </div>
  );
};
