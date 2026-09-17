/**
 * ChoiceGate - one-button-per-option picker for orchestrator choice gates
 * (e.g. PDF/DOCX/both, English/German/skip).
 *
 * Falls back to a reject button only when 'reject' is in allowed_actions.
 */

import type { OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { humanizeStep } from './gateFormat';

interface ChoiceGateProps {
  gate: OrchestratorGatePayload;
  isLoading: boolean;
  onSubmit: (
    action: OrchestratorGateAction,
    opts?: { feedback?: string; choice?: string }
  ) => void | Promise<void>;
}

const PRETTY_LABELS: Record<string, string> = {
  english: 'English',
  german: 'German (Anschreiben)',
  skip: 'Skip',
};

export const ChoiceGate = ({ gate, isLoading, onSubmit }: ChoiceGateProps) => {
  const choices = gate.choices || [];
  const allowsReject = gate.allowed_actions.includes('reject');

  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-border bg-primary-surface/25 p-4 shadow-sm sm:p-5">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-meta font-semibold uppercase tracking-[0.14em] text-primary-text">
          Pick one
        </span>
        <span className="text-meta font-medium text-ink-faint">{humanizeStep(gate.step)}</span>
      </div>

      {gate.narration && (
        <p className="mb-3 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
      )}

      <div className="flex flex-wrap gap-2">
        {choices.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => onSubmit('choose', { choice: c })}
            disabled={isLoading}
            className="rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-fg transition hover:bg-primary-hover active:scale-[0.97] disabled:opacity-50"
          >
            {PRETTY_LABELS[c] ?? c}
          </button>
        ))}
        {allowsReject && (
          <button
            type="button"
            onClick={() => onSubmit('reject')}
            disabled={isLoading}
            className="rounded-full border border-border-strong px-4 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink disabled:opacity-50"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
