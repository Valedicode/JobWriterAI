/**
 * DocumentEditor - the finished documents, edited as they print.
 *
 * Shows the CV or the cover letter on an A4 sheet in its PDF's layout, with
 * the editable parts edited in place, or the generated PDF itself. Saving sends
 * only what the user changed, for the revision on screen; the backend
 * generates the PDF again and the session's reply lands in the chat. A
 * document that changed meanwhile (an approved chat edit) is reloaded when
 * there are no unsaved edits, and otherwise reported instead of overwritten.
 */

import { useState } from 'react';
import { useDocuments } from '@/hooks/useDocuments';
import type { DocumentSaveResponse } from '@/types';
import { CoverLetterEditor } from './CoverLetterEditor';
import { CvEditor } from './CvEditor';

export type RunDocumentSave = (
  echo: string,
  save: (sessionId: string) => Promise<DocumentSaveResponse>
) => Promise<DocumentSaveResponse | null>;

type Kind = 'cv' | 'cover_letter';
const NAMES: Record<Kind, string> = { cv: 'CV', cover_letter: 'Cover letter' };

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
  const [chosen, setChosen] = useState<Kind>('cv');

  if (!documents) {
    return (
      <div className="flex-1 px-6 py-6 text-sm text-ink-muted" role="status">
        {error ?? 'Loading your documents…'}
      </div>
    );
  }
  const kinds = (['cv', 'cover_letter'] as const).filter((k) => documents[k]);
  if (kinds.length === 0) {
    return (
      <div className="flex-1 px-6 py-6">
        <h2 className="text-lg font-semibold text-ink">Nothing to edit yet</h2>
        <p className="mt-1 text-sm text-ink-muted">Your CV and cover letter can be edited here once their PDFs exist.</p>
      </div>
    );
  }
  const kind = kinds.includes(chosen) ? chosen : kinds[0];

  const title =
    kinds.length > 1 ? (
      <div role="tablist" aria-label="Document" className="mr-1 flex gap-1">
        {kinds.map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={k === kind}
            onClick={() => setChosen(k)}
            className={`rounded-md px-2 py-1 text-sm font-semibold transition-colors ${
              k === kind ? 'text-ink underline decoration-primary decoration-2 underline-offset-[6px]' : 'text-ink-muted hover:text-ink'
            }`}
          >
            {NAMES[k]}
          </button>
        ))}
      </div>
    ) : (
      <h2 className="mr-1 text-sm font-semibold text-ink">{NAMES[kind]}</h2>
    );

  const shared = { sessionId, title, editable: documents.editable, busy, runDocumentSave, onDocuments: setDocuments };
  // Each document keeps its own draft; switching documents keeps both mounted
  // would cost little, but a switch with unsaved edits is rare, so the editor
  // of the other document starts fresh from its saved version.
  return kind === 'cv' && documents.cv ? (
    <CvEditor key="cv" view={documents.cv} {...shared} />
  ) : documents.cover_letter ? (
    <CoverLetterEditor key="cover_letter" view={documents.cover_letter} {...shared} />
  ) : null;
};
