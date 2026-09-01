/**
 * InputGate - single text field for orchestrator input gates
 * (e.g. cover letter recipient when not found in the job posting).
 */

import { useState } from 'react';
import type { OrchestratorGateAction, OrchestratorGatePayload } from '@/types';

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
    <div className="mx-auto max-w-3xl rounded-xl border border-primary/30 bg-primary-surface/50 p-4 shadow-sm">
      <div className="mb-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-primary-text">
          Input needed · {gate.step}
        </div>
        <p className="mt-1 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
      </div>

      <div className="space-y-2">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Recipient name"
          className="w-full rounded border border-border-strong bg-surface px-3 py-2 text-[16px] text-ink placeholder:text-ink-faint sm:text-sm"
          maxLength={200}
          disabled={isLoading}
        />
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading || !value.trim()}
          className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-50"
        >
          Continue
        </button>
      </div>
    </div>
  );
};
