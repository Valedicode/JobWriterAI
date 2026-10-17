/**
 * useDocuments - the session's finished documents for the editor
 * (GET /api/documents/{session}).
 *
 * Fetched when the editor opens and again whenever `refreshKey` changes, i.e.
 * after every reply from the session: an approved chat edit changes the CV, a
 * re-run can drop the cover letter. A response that arrives after a newer
 * request started is dropped.
 */

import { useEffect, useState } from 'react';
import { getDocuments, getErrorMessage } from '@/lib/api';
import type { DocumentsResponse } from '@/types';

export const useDocuments = (sessionId: string, refreshKey: number) => {
  const [documents, setDocuments] = useState<DocumentsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let current = true;
    getDocuments(sessionId).then(
      (next) => {
        if (!current) return;
        setDocuments(next);
        setError(null);
      },
      (err) => {
        if (current) setError(getErrorMessage(err));
      }
    );
    return () => {
      current = false;
    };
  }, [sessionId, refreshKey]);

  return { documents, setDocuments, error };
};
