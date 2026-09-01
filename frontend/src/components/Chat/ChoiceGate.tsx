/**
 * ChoiceGate - one-button-per-option picker for orchestrator choice gates
 * (e.g. PDF/DOCX/both, English/German/skip).
 *
 * Falls back to a reject button only when 'reject' is in allowed_actions.
 */

import type { OrchestratorGateAction, OrchestratorGatePayload } from '@/types';

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
    <div className="mx-auto max-w-3xl rounded-xl border border-warning-border bg-warning-surface/60 p-4 shadow-sm">
      <div className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-warning-text">
          Pick one · {gate.step}
        </div>
        <p className="mt-1 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {choices.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => onSubmit('choose', { choice: c })}
            disabled={isLoading}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-50"
          >
            {PRETTY_LABELS[c] ?? c}
          </button>
        ))}
        {allowsReject && (
          <button
            type="button"
            onClick={() => onSubmit('reject')}
            disabled={isLoading}
            className="rounded-md border border-danger-border bg-surface px-3 py-1.5 text-sm font-medium text-danger-text hover:bg-danger-surface disabled:opacity-50"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};
