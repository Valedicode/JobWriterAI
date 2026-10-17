/**
 * CoverLetterEditor - the finished cover letter's draft: the editable fields
 * as the user changes them, the salutation for a new recipient (asked from the
 * server, which applies the PDF's rules), and saving only what changed.
 */

import { useEffect, useState } from 'react';
import { getErrorMessage, resetCoverLetter, saveCoverLetter, viewCoverLetter } from '@/lib/api';
import type { CoverLetterPrint, CoverLetterView, DocumentSaveResponse, DocumentsResponse } from '@/types';
import { CoverLetterSheet } from './CoverLetterSheet';
import { EditorFrame } from './EditorFrame';
import type { RunDocumentSave } from './DocumentEditor';

const changedFields = (draft: Record<string, string>, base: Record<string, string>) =>
  Object.fromEntries(Object.entries(draft).filter(([k, v]) => v !== (base[k] ?? '')));

interface CoverLetterEditorProps {
  sessionId: string;
  title: React.ReactNode;
  view: CoverLetterView;
  editable: boolean;
  busy: boolean;
  runDocumentSave: RunDocumentSave;
  onDocuments: (documents: DocumentsResponse) => void;
}

export const CoverLetterEditor = ({
  sessionId,
  title,
  view,
  editable,
  busy,
  runDocumentSave,
  onDocuments,
}: CoverLetterEditorProps) => {
  // The revision the draft was started from, and the draft itself.
  const [base, setBase] = useState({ revision: view.revision, fields: view.fields, letter: view.letter });
  const [draft, setDraft] = useState(view.fields);
  // The salutation and sign-off for the recipient in the draft (from the server).
  const [print, setPrint] = useState<CoverLetterPrint>(view.letter);
  const [saving, setSaving] = useState(false);
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

  const locked = !editable || busy || saving;

  const run = async (echo: string, save: (sessionId: string) => Promise<DocumentSaveResponse>) => {
    setSaving(true);
    setError(null);
    try {
      const result = await runDocumentSave(echo, save);
      if (!result) return;
      onDocuments(result.documents);
      const next = result.documents.cover_letter;
      if (next) {
        setBase({ revision: next.revision, fields: next.fields, letter: next.letter });
        setDraft(next.fields);
        setPrint(next.letter);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <EditorFrame
      title={title}
      noun="letter"
      view={view}
      editable={editable}
      dirty={dirty}
      moved={moved}
      locked={locked}
      saving={saving}
      error={error}
      onSave={() =>
        run('Saved my edits to the cover letter.', (sid) =>
          saveCoverLetter(sid, { revision: base.revision, fields: changes })
        )
      }
      onReset={() =>
        run('Went back to the generated cover letter.', (sid) => resetCoverLetter(sid, { revision: base.revision }))
      }
      onDiscard={() => {
        setDraft(base.fields);
        setPrint(base.letter);
        setError(null);
      }}
    >
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
    </EditorFrame>
  );
};
