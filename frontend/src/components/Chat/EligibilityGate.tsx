/**
 * EligibilityGate - answers for the confirm_eligibility gate.
 *
 * The posting states a must-have (a language level, enrollment, a degree)
 * that the CV says nothing about. Each question gets a level picker or a
 * short text field; "Save answers" sends them as `answers` keyed by question
 * id, "Skip" continues without them (the items stay out of the score).
 */

import { useState } from 'react';
import type {
  BulletDecision,
  EligibilityQuestion,
  OrchestratorGateAction,
  OrchestratorGatePayload,
} from '@/types';
import { humanizeStep } from './gateFormat';

interface EligibilityGateProps {
  gate: OrchestratorGatePayload;
  isLoading: boolean;
  onSubmit: (
    action: OrchestratorGateAction,
    opts?: {
      feedback?: string;
      choice?: string;
      decisions?: Record<string, BulletDecision>;
      answers?: Record<string, string>;
    }
  ) => void | Promise<void>;
}

const OPTION_LABELS: Record<string, string> = {
  native: 'Native',
  none: "I don't speak it",
};

const fieldClass =
  'w-full rounded-full border border-border-strong bg-surface px-4 py-2 text-[16px] text-ink placeholder:text-ink-faint focus:border-primary focus:outline-none disabled:opacity-50 sm:text-sm';

function asQuestions(preview: unknown): EligibilityQuestion[] {
  const raw = (preview as { questions?: unknown } | null)?.questions;
  return Array.isArray(raw)
    ? raw.filter((q): q is EligibilityQuestion => Boolean(q && typeof q.id === 'string' && q.prompt))
    : [];
}

export const EligibilityGate = ({ gate, isLoading, onSubmit }: EligibilityGateProps) => {
  const questions = asQuestions(gate.preview);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const filled = Object.fromEntries(
    Object.entries(answers).filter(([, value]) => value.trim().length > 0)
  );
  const filledCount = Object.keys(filled).length;

  const handleSave = async () => {
    if (filledCount === 0) return;
    // `feedback` is required for an 'edit' resolution; it doubles as a
    // readable summary in the transcript.
    const summary = questions
      .filter((q) => filled[q.id])
      .map((q) => `${q.requirement}: ${filled[q.id]}`)
      .join('; ');
    await onSubmit('edit', { feedback: summary, answers: filled });
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
        <p className="mb-4 text-body text-ink [overflow-wrap:anywhere]">{gate.narration}</p>
      )}

      <div className="mb-4 space-y-3.5">
        {questions.map((q) => {
          const fieldId = `eligibility-${q.id}`;
          return (
            <div key={q.id}>
              <label htmlFor={fieldId} className="mb-1 block text-sm font-medium text-ink">
                {q.prompt}
              </label>
              <p className="mb-1.5 text-meta text-ink-faint [overflow-wrap:anywhere]">
                Posting asks for: {q.requirement}
              </p>
              {q.options && q.options.length > 0 ? (
                <select
                  id={fieldId}
                  value={answers[q.id] ?? ''}
                  onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                  disabled={isLoading}
                  className={fieldClass}
                >
                  <option value="">Not answered</option>
                  {q.options.map((option) => (
                    <option key={option} value={option}>
                      {OPTION_LABELS[option] ?? option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  id={fieldId}
                  type="text"
                  value={answers[q.id] ?? ''}
                  onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                  placeholder={q.placeholder ?? ''}
                  maxLength={200}
                  disabled={isLoading}
                  className={fieldClass}
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleSave}
          disabled={isLoading || filledCount === 0}
          className="rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-fg transition hover:bg-primary-hover active:scale-[0.97] disabled:opacity-50"
        >
          {filledCount > 0 ? `Save ${filledCount} answer${filledCount === 1 ? '' : 's'}` : 'Save answers'}
        </button>
        <button
          type="button"
          onClick={() => onSubmit('approve')}
          disabled={isLoading}
          className="rounded-full border border-border-strong px-4 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-surface-sunken hover:text-ink disabled:opacity-50"
        >
          Skip
        </button>
      </div>
    </div>
  );
};
