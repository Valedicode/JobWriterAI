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
import { humanizeStep } from './gateFormat';
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
    <div className="mx-auto max-w-3xl rounded-2xl border border-border bg-primary-surface/25 p-4 shadow-sm sm:p-5">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-meta font-semibold uppercase tracking-[0.14em] text-primary-text">
          Approval needed
        </span>
        <span className="text-meta font-medium text-ink-faint">{humanizeStep(gate.step)}</span>
      </div>

      <div className="mb-4 rounded-xl border border-border bg-surface p-4">
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
        <div className="space-y-2.5">
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="What should change? (e.g. “Use stronger action verbs”, “Mention Kubernetes more”)"
            className="w-full rounded-xl border border-border-strong bg-surface px-3.5 py-2.5 text-[16px] text-ink placeholder:text-ink-faint focus:border-primary focus:outline-none sm:text-sm"
            rows={3}
            maxLength={4000}
            disabled={isLoading}
            autoFocus
          />
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleEditSubmit}
              disabled={isLoading || !feedback.trim()}
              className="rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-fg transition hover:bg-primary-hover active:scale-[0.97] disabled:opacity-50"
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
              className="rounded-full border border-border-strong px-4 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-surface-sunken disabled:opacity-50"
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
              className="rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-fg transition hover:bg-primary-hover active:scale-[0.97] disabled:opacity-50"
            >
              Approve
            </button>
          )}
          {allows('edit') && (
            <button
              type="button"
              onClick={() => setShowEdit(true)}
              disabled={isLoading}
              className="rounded-full border border-border-strong px-4 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-surface-sunken disabled:opacity-50"
            >
              Edit with feedback
            </button>
          )}
          {allows('reject') && (
            <button
              type="button"
              onClick={() => onSubmit('reject')}
              disabled={isLoading}
              className="rounded-full border border-border-strong px-4 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink disabled:opacity-50"
            >
              Reject
            </button>
          )}
        </div>
      )}
    </div>
  );
};
