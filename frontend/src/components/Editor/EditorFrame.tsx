/**
 * EditorFrame - what the CV and cover letter editors share: the toolbar
 * (Edit / PDF, the save state, Save, Discard, Reset to generated, Download),
 * the notices, and the body: the editable sheet, or the generated PDF.
 * Ctrl/Cmd+S saves.
 */

import { useState } from 'react';
import { downloadFile, getErrorMessage, triggerFileDownload } from '@/lib/api';

type Mode = 'edit' | 'pdf';

/** What the frame needs of a document view. */
interface FramedView {
  revision: number;
  rendered_revision: number | null;
  edited: boolean;
  filename: string;
  download_url: string | null;
}

interface EditorFrameProps {
  /** The heading: the document's name, or the switcher between documents. */
  title: React.ReactNode;
  /** 'letter' or 'CV', for the status line. */
  noun: string;
  view: FramedView;
  /** False while a change proposed in the chat waits for confirmation. */
  editable: boolean;
  dirty: boolean;
  /** The document changed elsewhere since editing started. */
  moved: boolean;
  /** Editing is paused (the chat is busy, a save is running, or not editable). */
  locked: boolean;
  saving: boolean;
  error: string | null;
  onSave: () => void;
  onDiscard: () => void;
  onReset: () => void;
  /** The editable sheet. */
  children: React.ReactNode;
}

export const EditorFrame = ({
  title,
  noun,
  view,
  editable,
  dirty,
  moved,
  locked,
  saving,
  error,
  onSave,
  onDiscard,
  onReset,
  children,
}: EditorFrameProps) => {
  const [mode, setMode] = useState<Mode>('edit');
  const [confirmReset, setConfirmReset] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const pdfOutdated = view.rendered_revision !== view.revision;
  const canSave = !locked && !moved && (dirty || pdfOutdated);

  const download = async () => {
    if (!view.download_url) return;
    setDownloadError(null);
    try {
      triggerFileDownload(await downloadFile(view.download_url), view.filename);
    } catch (err) {
      setDownloadError(getErrorMessage(err));
    }
  };

  const status = saving
    ? 'Saving and generating the PDF…'
    : moved
      ? `This ${noun} changed in the chat since you started editing.`
      : dirty
        ? 'Unsaved changes'
        : pdfOutdated
          ? 'Saved, but the PDF could not be generated. Save again to retry.'
          : view.edited
            ? 'Saved with your edits'
            : 'As generated';

  return (
    <div
      className="flex min-h-0 flex-1 flex-col"
      onKeyDown={(e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
          e.preventDefault();
          if (canSave) onSave();
        }
      }}
    >
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-6 py-3">
        {title}
        <div role="radiogroup" aria-label="View" className="flex rounded-full border border-border-strong p-0.5">
          {(['edit', 'pdf'] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              onClick={() => setMode(m)}
              className={`rounded-full px-3 py-1 text-meta font-medium transition-colors ${
                mode === m ? 'bg-primary text-primary-fg' : 'text-ink-muted hover:text-ink'
              }`}
            >
              {m === 'edit' ? 'Edit' : 'PDF'}
            </button>
          ))}
        </div>
        <p role="status" aria-live="polite" className="min-w-0 flex-1 truncate text-meta text-ink-muted">
          {status}
        </p>
        <div className="flex items-center gap-2">
          {moved && dirty && (
            <button type="button" onClick={onDiscard} className={secondaryButton}>
              Load the new version
            </button>
          )}
          {dirty && !moved && (
            <button type="button" onClick={onDiscard} disabled={locked} className={secondaryButton}>
              Discard
            </button>
          )}
          {view.edited && !dirty && !confirmReset && (
            <button type="button" onClick={() => setConfirmReset(true)} disabled={locked} className={secondaryButton}>
              Reset to generated
            </button>
          )}
          {confirmReset && (
            <span className="flex items-center gap-2 text-meta text-ink">
              Discard your edits?
              <button
                type="button"
                onClick={() => {
                  setConfirmReset(false);
                  onReset();
                }}
                className={secondaryButton}
              >
                Reset
              </button>
              <button type="button" onClick={() => setConfirmReset(false)} className={secondaryButton}>
                Keep
              </button>
            </span>
          )}
          {(dirty || pdfOutdated) && (
            <button type="button" onClick={onSave} disabled={!canSave} className={primaryButton}>
              {saving ? 'Saving…' : 'Save'}
            </button>
          )}
          {!dirty && view.download_url && (
            <button type="button" onClick={download} className={secondaryButton}>
              Download PDF
            </button>
          )}
        </div>
      </div>

      {!editable && (
        <p className="shrink-0 border-b border-warning-border bg-warning-surface px-6 py-2 text-meta text-warning-fg">
          A change is waiting for your confirmation in the chat. Confirm or cancel it to edit your documents.
        </p>
      )}
      {(error || downloadError) && (
        <p role="alert" className="shrink-0 border-b border-border px-6 py-2 text-meta text-danger-text">
          {error || downloadError}
        </p>
      )}

      {mode === 'edit' ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">{children}</div>
      ) : (
        <PdfPreview url={view.download_url} revision={view.rendered_revision} title={`PDF of ${view.filename}`} />
      )}
    </div>
  );
};

const PdfPreview = ({ url, revision, title }: { url: string | null; revision: number | null; title: string }) =>
  url ? (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* The revision in the URL makes the frame load the PDF again after a save. */}
      <iframe src={`${url}?inline=true&v=${revision ?? 0}`} title={title} className="min-h-[60vh] w-full flex-1 bg-white" />
      <a
        href={`${url}?inline=true`}
        target="_blank"
        rel="noreferrer"
        className="shrink-0 px-6 py-2 text-meta font-medium text-primary-text hover:underline"
      >
        Open the PDF in a new tab
      </a>
    </div>
  ) : (
    <p className="flex-1 px-6 py-6 text-sm text-ink-muted">The PDF isn&apos;t available.</p>
  );

const secondaryButton =
  'rounded-full border border-border-strong px-3 py-1 text-meta font-medium text-ink transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50';
const primaryButton =
  'rounded-full bg-primary px-4 py-1 text-meta font-semibold text-primary-fg transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-50';
