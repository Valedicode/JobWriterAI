/**
 * ApprovalGate - approve / edit / reject buttons for orchestrator approval gates.
 *
 * Edit reveals an inline textarea; submitting it calls onSubmit('edit', {feedback}).
 * Buttons hide themselves if not in `allowed_actions` so the same component
 * serves both soft gates (approve/reject only) and hard gates (approve/edit/reject).
 */

import { useState } from 'react';
import type { BulletDecision, OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { PreviewCard } from './PreviewCard';
import { humanizeStep } from './gateFormat';
import { ErrorBoundary } from '@/components/ErrorBoundary';

interface ApprovalGateProps {
  gate: OrchestratorGatePayload;
  isLoading: boolean;
  onSubmit: (
    action: OrchestratorGateAction,
    opts?: { feedback?: string; choice?: string; decisions?: Record<string, BulletDecision> }
  ) => void | Promise<void>;
}

export const ApprovalGate = ({ gate, isLoading, onSubmit }: ApprovalGateProps) => {
  const [showEdit, setShowEdit] = useState(false);
  const [feedback, setFeedback] = useState('');
  // Per-bullet decisions for approve_selection / approve_rewrite, keyed by
  // bullet_id. Only overrides from the default (include / accept) are
  // stored, so a plain Approve with no interaction sends an empty map and
  // behaves exactly as before this feature existed.
  const [decisions, setDecisions] = useState<Record<string, BulletDecision>>({});

  const allows = (a: OrchestratorGateAction) => gate.allowed_actions.includes(a);
  const decisionCount = Object.keys(decisions).length;
  // Steps where approve / reject mean something more specific than the words.
  const stepLabels: Record<string, { approve: string; reject: string }> = {
    step_failed: { approve: 'Retry', reject: 'Stop' }, // retry the failed step, or stop
    confirm_change: { approve: 'Confirm', reject: 'Cancel' }, // a re-run / profile update from the chat
    verifier_failed: { approve: 'Keep my original wording', reject: 'Reject' }, // blocked rewrites
    approve_translation: { approve: 'Use translation', reject: 'Keep my CV as it is' },
  };
  const labels = stepLabels[gate.step];
  const approveLabel =
    labels?.approve ?? (decisionCount > 0 ? `Apply (${decisionCount} customized)` : 'Approve');
  const rejectLabel = labels?.reject ?? 'Reject';

  const handleDecisionChange = (bulletId: string, decision: BulletDecision | null) => {
    setDecisions((prev) => {
      const next = { ...prev };
      if (decision === null) {
        delete next[bulletId];
      } else {
        next[bulletId] = decision;
      }
      return next;
    });
  };

  const handleEditSubmit = async () => {
    if (!feedback.trim()) return;
    await onSubmit('edit', { feedback: feedback.trim(), decisions });
    setShowEdit(false);
    setFeedback('');
    setDecisions({});
  };

  const handleApprove = async () => {
    await onSubmit('approve', { decisions });
    setDecisions({});
  };

  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-border bg-primary-surface/25 p-4 shadow-sm sm:p-5 lg:mx-0 lg:flex lg:h-full lg:min-h-0 lg:max-w-none lg:flex-col lg:rounded-none lg:border-0 lg:bg-transparent lg:p-0 lg:shadow-none">
      {/* Scrolls on its own from lg; the action bar below stays in view. */}
      <div className="lg:min-h-0 lg:flex-1 lg:overflow-y-auto lg:px-6 lg:py-5">
      <div className="mb-3 flex shrink-0 items-center gap-2 lg:hidden">
        <span className="text-meta font-semibold uppercase tracking-[0.14em] text-primary-text">
          Approval needed
        </span>
        <span className="text-meta font-medium text-ink-faint">{humanizeStep(gate.step)}</span>
      </div>
      <div className="mb-4 hidden lg:block">
        <h2 className="text-lg font-semibold text-ink">{humanizeStep(gate.step)}</h2>
        <p className="mt-0.5 text-sm text-ink-muted">Check the details, then decide below.</p>
      </div>

      <div className="mb-4 rounded-xl border border-border bg-surface p-4 lg:mb-0 lg:border-0 lg:bg-transparent lg:p-0">
        <ErrorBoundary
          fallback={
            <p className="text-sm text-ink-muted">
              This preview couldn’t be rendered. You can still approve, edit, or reject below.
            </p>
          }
        >
          <PreviewCard gate={gate} decisions={decisions} onDecisionChange={handleDecisionChange} />
        </ErrorBoundary>
      </div>
      </div>

      <div className="lg:shrink-0 lg:border-t lg:border-border lg:bg-surface lg:px-6 lg:py-4">
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
          {decisionCount > 0 && gate.step === 'approve_rewrite' && (
            <p className="text-meta text-ink-faint">
              {decisionCount} bullet{decisionCount === 1 ? '' : 's'} marked above will be kept as-is; only the
              rest are re-run with your feedback.
            </p>
          )}
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
              onClick={handleApprove}
              disabled={isLoading}
              className="rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-fg transition hover:bg-primary-hover active:scale-[0.97] disabled:opacity-50"
            >
              {approveLabel}
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
              {rejectLabel}
            </button>
          )}
        </div>
      )}
      </div>
    </div>
  );
};
