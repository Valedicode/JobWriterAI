/**
 * DocumentEditor - the finished documents, edited as they print.
 *
 * Shows a document on an A4 sheet in its PDF's layout with the editable parts
 * edited in place, or the generated PDF itself. Saving sends only what the
 * user changed, for the revision on screen; the backend generates the PDF
 * again and the session's reply lands in the chat. A document that changed
 * meanwhile (an approved chat edit) is reloaded when there are no unsaved
 * edits, and otherwise reported instead of overwritten.
 */

import { useEffect, useState } from 'react';
import {
  downloadFile,
  getErrorMessage,
  resetCoverLetter,
  saveCoverLetter,
  triggerFileDownload,
  viewCoverLetter,
} from '@/lib/api';
import { useDocuments } from '@/hooks/useDocuments';
import type { CoverLetterPrint, CoverLetterView, DocumentSaveResponse, DocumentsResponse } from '@/types';
import { CoverLetterSheet } from './CoverLetterSheet';

type Mode = 'edit' | 'pdf';

export type RunDocumentSave = (
  echo: string,
  save: (sessionId: string) => Promise<DocumentSaveResponse>
) => Promise<DocumentSaveResponse | null>;

interface DocumentEditorProps {
  sessionId: string;
  /** Changes after every reply from the session, so the documents are fetched again. */
  refreshKey: number;
  /** True while the chat is waiting for the session. */
  busy: boolean;
  runDocumentSave: RunDocumentSave;
}

export const DocumentEditor = ({ sessionId, refreshKey, busy, runDocumentSave }: DocumentEditorProps) => {
  const { documents, setDocuments, error } = useDocuments(sessionId, refreshKey);

  if (!documents) {
    return (
      <div className="flex-1 px-6 py-6 text-sm text-ink-muted" role="status">
        {error ?? 'Loading your documents…'}
      </div>
    );
  }
  if (!documents.cover_letter) {
    return (
      <div className="flex-1 px-6 py-6">
        <h2 className="text-lg font-semibold text-ink">Nothing to edit yet</h2>
        <p className="mt-1 text-sm text-ink-muted">Your cover letter can be edited here once its PDF exists.</p>
      </div>
    );
  }
  return (
    <CoverLetterEditor
      sessionId={sessionId}
      view={documents.cover_letter}
      editable={documents.editable}
      busy={busy}
      runDocumentSave={runDocumentSave}
      onDocuments={setDocuments}
    />
  );
};

const changedFields = (draft: Record<string, string>, base: Record<string, string>) =>
  Object.fromEntries(Object.entries(draft).filter(([k, v]) => v !== (base[k] ?? '')));

interface CoverLetterEditorProps {
  sessionId: string;
  view: CoverLetterView;
  editable: boolean;
  busy: boolean;
  runDocumentSave: RunDocumentSave;
  onDocuments: (documents: DocumentsResponse) => void;
}

const CoverLetterEditor = ({ sessionId, view, editable, busy, runDocumentSave, onDocuments }: CoverLetterEditorProps) => {
  const [mode, setMode] = useState<Mode>('edit');
  // The revision the draft was started from, and the draft itself.
  const [base, setBase] = useState({ revision: view.revision, fields: view.fields, letter: view.letter });
  const [draft, setDraft] = useState(view.fields);
  // The salutation and sign-off for the recipient in the draft (from the server).
  const [print, setPrint] = useState<CoverLetterPrint>(view.letter);
  const [saving, setSaving] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const changes = changedFields(draft, base.fields);
  const dirty = Object.keys(changes).length > 0;

  // The letter changed elsewhere since the draft started. Without unsaved
  // edits, take the new version (adjusted while rendering, not in an effect).
  const moved = view.revision !== base.revision;
  if (moved && !dirty) {
    setBase({ revision: view.revision, fields: view.fields, letter: view.letter });
    setDraft(view.fields);
    setPrint(view.letter);
  }

  // A new recipient changes the salutation (and an English letter's sign-off):
  // ask the server, which applies the same rules as the PDF, once typing pauses.
  const recipient = draft.recipient ?? '';
  useEffect(() => {
    if (recipient === base.fields.recipient || !recipient.trim()) return;
    const timer = setTimeout(async () => {
      try {
        const next = await viewCoverLetter(sessionId, { revision: base.revision, fields: { recipient } });
        setPrint(next.letter);
      } catch {
        // The salutation updates on save; a failed preview changes nothing.
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [recipient, base, sessionId]);
  const shownPrint = recipient === base.fields.recipient ? base.letter : print;
  // An untouched sign-off is the one the recipient gets ("Yours faithfully"
  // to a team), so it follows the preview; once edited, it is the user's.
  const shownFields =
    draft.sign_off === base.fields.sign_off ? { ...draft, sign_off: shownPrint.sign_off } : draft;

  const adopt = (result: DocumentSaveResponse | null) => {
    if (!result) return;
    onDocuments(result.documents);
    const next = result.documents.cover_letter;
    if (next) {
      setBase({ revision: next.revision, fields: next.fields, letter: next.letter });
      setDraft(next.fields);
      setPrint(next.letter);
    }
  };

  const pdfOutdated = view.rendered_revision !== view.revision;
  const locked = !editable || busy || saving;
  const canSave = !locked && !moved && (dirty || pdfOutdated);

  const save = async () => {
    if (!canSave) return;
    setSaving(true);
    setError(null);
    try {
      adopt(await runDocumentSave('Saved my edits to the cover letter.', (sid) =>
        saveCoverLetter(sid, { revision: base.revision, fields: changes })
      ));
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setConfirmReset(false);
    setSaving(true);
    setError(null);
    try {
      adopt(await runDocumentSave('Went back to the generated cover letter.', (sid) =>
        resetCoverLetter(sid, { revision: base.revision })
      ));
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  const discard = () => {
    setDraft(base.fields);
    setPrint(base.letter);
    setError(null);
  };

  const download = async () => {
    if (!view.download_url) return;
    try {
      triggerFileDownload(await downloadFile(view.download_url), view.filename);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const status = saving
    ? 'Saving and generating the PDF…'
    : moved
      ? 'This letter changed in the chat since you started editing.'
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
          void save();
        }
      }}
    >
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-6 py-3">
        <h2 className="mr-1 text-sm font-semibold text-ink">Cover letter</h2>
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
            <button type="button" onClick={discard} className={secondaryButton}>
              Load the new version
            </button>
          )}
          {dirty && !moved && (
            <button type="button" onClick={discard} disabled={locked} className={secondaryButton}>
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
              <button type="button" onClick={reset} className={secondaryButton}>
                Reset
              </button>
              <button type="button" onClick={() => setConfirmReset(false)} className={secondaryButton}>
                Keep
              </button>
            </span>
          )}
          {(dirty || pdfOutdated) && (
            <button type="button" onClick={save} disabled={!canSave} className={primaryButton}>
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
      {error && (
        <p role="alert" className="shrink-0 border-b border-border px-6 py-2 text-meta text-danger-text">
          {error}
        </p>
      )}

      {mode === 'edit' ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
          <label className="mx-auto mb-4 flex max-w-[52rem] flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink">
            <span className="font-medium">Addressed to</span>
            <input
              type="text"
              value={recipient}
              disabled={locked}
              onChange={(e) => setDraft({ ...draft, recipient: e.target.value })}
              className="min-w-[14rem] flex-1 rounded-lg border border-border-strong bg-surface px-3 py-1.5 text-sm text-ink focus:border-primary focus:outline-none disabled:opacity-60"
            />
            <span className="w-full text-meta text-ink-muted">The salutation follows from it.</span>
          </label>
          <CoverLetterSheet
            letter={shownPrint}
            fields={shownFields}
            onChange={(field, value) => setDraft({ ...draft, [field]: value })}
            disabled={locked}
          />
        </div>
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
