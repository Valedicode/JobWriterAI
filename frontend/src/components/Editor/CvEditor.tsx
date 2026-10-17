/**
 * CvEditor - the finished CV's draft, kept as the list of edits the backend
 * applies in order (set a field, insert or delete a bullet).
 *
 * A text edit is added to the list when its field loses focus; the page
 * already shows it. Adding or removing a bullet changes which path each later
 * bullet has, so the view is fetched again with the edits so far
 * (POST .../cv/view): every path the page shows is one the next edit can use.
 */

import { useState } from 'react';
import { getErrorMessage, resetCv, saveCv, viewCv } from '@/lib/api';
import type { CvDocumentView, CvEdit, DocumentSaveResponse, DocumentsResponse } from '@/types';
import { CvSheet } from './CvSheet';
import { EditorFrame } from './EditorFrame';
import type { RunDocumentSave } from './DocumentEditor';

/** The edits with a new value for `path`: the earlier one replaced when no bullet was added or removed since. */
const withSet = (edits: CvEdit[], path: string, value: string): CvEdit[] => {
  const lastStructural = edits.findLastIndex((e) => e.op !== 'set');
  const earlier = edits.findLastIndex((e) => e.op === 'set' && e.path === path);
  if (earlier > lastStructural) {
    return edits.map((e, i) => (i === earlier ? { ...e, value } : e));
  }
  return [...edits, { op: 'set', path, value }];
};

interface CvEditorProps {
  sessionId: string;
  title: React.ReactNode;
  view: CvDocumentView;
  editable: boolean;
  busy: boolean;
  runDocumentSave: RunDocumentSave;
  onDocuments: (documents: DocumentsResponse) => void;
}

export const CvEditor = ({ sessionId, title, view, editable, busy, runDocumentSave, onDocuments }: CvEditorProps) => {
  // The saved version the edits apply to, the edits, and the page as they leave it.
  const [base, setBase] = useState(view);
  const [edits, setEdits] = useState<CvEdit[]>([]);
  const [shown, setShown] = useState(view);
  // Bumped to show a view afresh (the page holds typed text until then).
  const [version, setVersion] = useState(0);
  const [saving, setSaving] = useState(false);
  const [restructuring, setRestructuring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = edits.length > 0;
  // The CV changed elsewhere (an approved chat edit) since editing started.
  // Without unsaved edits, take the new version (adjusted while rendering).
  const moved = view.revision !== base.revision;
  if (moved && !dirty) {
    setBase(view);
    setShown(view);
    setVersion(version + 1);
  }

  const locked = !editable || busy || saving || restructuring;

  const restructure = async (edit: CvEdit) => {
    const next = [...edits, edit];
    setRestructuring(true);
    setError(null);
    try {
      setShown(await viewCv(sessionId, { revision: base.revision, edits: next }));
      setEdits(next);
      setVersion(version + 1);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setRestructuring(false);
    }
  };

  const run = async (echo: string, save: (sessionId: string) => Promise<DocumentSaveResponse>) => {
    setSaving(true);
    setError(null);
    try {
      const result = await runDocumentSave(echo, save);
      if (!result) return;
      onDocuments(result.documents);
      const next = result.documents.cv;
      if (next) {
        setBase(next);
        setShown(next);
        setEdits([]);
        setVersion(version + 1);
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
      noun="CV"
      view={view}
      editable={editable}
      dirty={dirty}
      moved={moved}
      locked={locked}
      saving={saving}
      error={error}
      onSave={() => run('Saved my edits to the CV.', (sid) => saveCv(sid, { revision: base.revision, edits }))}
      onReset={() => run('Went back to the generated CV.', (sid) => resetCv(sid, { revision: base.revision }))}
      onDiscard={() => {
        setEdits([]);
        setShown(base);
        setVersion(version + 1);
        setError(null);
      }}
    >
      <p className="mx-auto mb-3 max-w-[52rem] text-meta text-ink-muted">
        Click any text from your CV to change it. Hover a bullet to add or remove one.
      </p>
      <CvSheet
        key={version}
        view={shown}
        disabled={locked}
        onSet={(path, value) => setEdits((current) => withSet(current, path, value))}
        onInsert={(listPath, index) => restructure({ op: 'insert', path: listPath, index, value: 'New bullet' })}
        onDelete={(path) => restructure({ op: 'delete', path })}
      />
    </EditorFrame>
  );
};
