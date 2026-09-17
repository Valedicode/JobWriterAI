/**
 * InputGate - single text field for orchestrator input gates
 * (e.g. cover letter recipient when not found in the job posting).
 */

import { useState } from 'react';
import type { OrchestratorGateAction, OrchestratorGatePayload } from '@/types';
import { humanizeStep } from './gateFormat';

interface InputGateProps {
  gate: OrchestratorGatePayload;
  isLoading: boolean;
  onSubmit: (
    action: OrchestratorGateAction,
    opts?: { feedback?: string; choice?: string }
  ) => void | Promise<void>;
}

export const InputGate = ({ gate, isLoading, onSubmit }: InputGateProps) => {
  const [value, setValue] = useState('');

  const handleSubmit = async () => {
    if (!value.trim()) return;
    await onSubmit('edit', { feedback: value.trim() });
    setValue('');
  };

  return (
    <div className="mx-auto max-w-3xl rounded-2xl border border-border bg-primary-surface/25 p-4 shadow-sm sm:p-5">
      <div className="mb-3 flex items-center gap-2">
        <span className="text-meta font-semibold uppercase tracking-[0.14em] text-primary-text">
          Input needed
        </span>
        <span className="text-meta font-medium text-ink-faint">{humanizeStep(gate.step)}</span>
      </div>

      {gate.narration && (
        <p className="mb-3 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
      )}

      <div className="flex flex-col gap-2.5 sm:flex-row">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSubmit();
          }}
          placeholder="Recipient name"
          className="flex-1 rounded-full border border-border-strong bg-surface px-4 py-2 text-[16px] text-ink placeholder:text-ink-faint focus:border-primary focus:outline-none sm:text-sm"
          maxLength={200}
          disabled={isLoading}
          autoFocus
        />
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading || !value.trim()}
          className="rounded-full bg-primary px-5 py-2 text-sm font-medium text-primary-fg transition-colors hover:bg-primary-hover disabled:opacity-50"
        >
          Continue
        </button>
      </div>
    </div>
  );
};
