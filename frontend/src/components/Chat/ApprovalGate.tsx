/**
 * ApprovalGate - approve / edit / reject buttons for orchestrator approval gates.
 *
 * Edit reveals an inline textarea; submitting it calls onSubmit('edit', {feedback}).
 * Buttons hide themselves if not in `allowed_actions` so the same component
 * serves both soft gates (approve/reject only) and hard gates (approve/edit/reject).
 */

import { useState } from 'react';
import type { OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { PreviewCard } from './PreviewCard';
import { ErrorBoundary } from '@/components/ErrorBoundary';

interface ApprovalGateProps {
  gate: OrchestratorGatePayload;
  isLoading: boolean;
  onSubmit: (
    action: OrchestratorGateAction,
    opts?: { feedback?: string; choice?: string }
  ) => void | Promise<void>;
}

export const ApprovalGate = ({ gate, isLoading, onSubmit }: ApprovalGateProps) => {
  const [showEdit, setShowEdit] = useState(false);
  const [feedback, setFeedback] = useState('');

  const allows = (a: OrchestratorGateAction) => gate.allowed_actions.includes(a);

  const handleEditSubmit = async () => {
    if (!feedback.trim()) return;
    await onSubmit('edit', { feedback: feedback.trim() });
    setShowEdit(false);
    setFeedback('');
  };

  return (
    <div className="mx-auto max-w-3xl rounded-xl border border-primary/30 bg-primary-surface/50 p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-primary-text">
            Approval needed · {gate.step}
          </div>
          <p className="mt-1 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
        </div>
      </div>

      <div className="mb-4 rounded-lg border border-border bg-surface p-3">
        <ErrorBoundary
          fallback={
            <p className="text-sm text-ink-muted">
              This preview couldn’t be rendered. You can still approve, edit, or reject below.
            </p>
          }
        >
          <PreviewCard gate={gate} />
        </ErrorBoundary>
      </div>

      {showEdit ? (
        <div className="space-y-2">
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="What should change? (e.g. 'Use stronger action verbs', 'Mention Kubernetes more')"
            className="w-full rounded border border-border-strong bg-surface px-3 py-2 text-[16px] text-ink placeholder:text-ink-faint sm:text-sm"
            rows={3}
            maxLength={4000}
            disabled={isLoading}
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleEditSubmit}
              disabled={isLoading || !feedback.trim()}
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-50"
            >
              Submit edit
            </button>
            <button
              type="button"
              onClick={() => {
                setShowEdit(false);
                setFeedback('');
              }}
              disabled={isLoading}
              className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-ink-muted hover:bg-surface-sunken hover:text-ink"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          {allows('approve') && (
            <button
              type="button"
              onClick={() => onSubmit('approve')}
              disabled={isLoading}
              className="rounded-md bg-success px-3 py-1.5 text-sm font-medium text-success-fg hover:brightness-105 disabled:opacity-50"
            >
              Approve
            </button>
          )}
          {allows('edit') && (
            <button
              type="button"
              onClick={() => setShowEdit(true)}
              disabled={isLoading}
              className="rounded-md border border-primary/40 bg-surface px-3 py-1.5 text-sm font-medium text-primary-text hover:bg-primary-surface disabled:opacity-50"
            >
              Edit with feedback
            </button>
          )}
          {allows('reject') && (
            <button
              type="button"
              onClick={() => onSubmit('reject')}
              disabled={isLoading}
              className="rounded-md border border-danger-border bg-surface px-3 py-1.5 text-sm font-medium text-danger-text hover:bg-danger-surface disabled:opacity-50"
            >
              Reject
            </button>
          )}
        </div>
      )}
    </div>
  );
};
