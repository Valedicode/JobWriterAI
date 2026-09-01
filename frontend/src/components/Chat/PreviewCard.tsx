/**
 * PreviewCard - typed renderer for the preview payload of an orchestrator gate.
 *
 * Switches on the gate's `step` to pick a sensible compact rendering. Falls
 * back to a JSON dump in a code block for unknown steps so nothing is hidden.
 */

import type { OrchestratorGatePayload } from '@/types';

type PreviewMap = Record<string, unknown>;

interface PreviewCardProps {
  gate: OrchestratorGatePayload;
}

/** Coerce a possibly-malformed backend field to a safe array. */
const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);

// --------------------------------------------------------------------------
// Step-specific preview shapes. The orchestrator backend is the source of
// truth; these are mirrors used only for rendering convenience.
// --------------------------------------------------------------------------

interface ScorePreview {
  aggregate_score?: number;
  level?: string;
  dimensions?: Array<{ name: string; score: number; weight: number }>;
  gap_analysis?: {
    matched_skills?: unknown[];
    transferable_skills?: unknown[];
    missing_skills?: unknown[];
  };
  interpretation?: string;
}

interface SelectionPreview {
  selected_bullets?: Array<{
    section: string;
    original_text: string;
    relevance_score: number;
  }>;
  section_order?: string[];
  sections_to_emphasize?: string[];
}

interface RewritePreview {
  rewritten_bullets?: Array<{
    original: string;
    rewritten: string;
    confidence: number;
    keywords_added?: string[];
  }>;
  keywords_inserted?: string[];
}

interface CoverLetterPreview {
  opening_paragraph?: string;
  body_paragraph_1?: string;
  body_paragraph_2?: string;
  body_paragraph_3?: string;
  closing_paragraph?: string;
  betreff?: string;
  grussformel?: string;
}

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="mb-3 last:mb-0">
    <h4 className="mb-1 text-meta font-semibold uppercase tracking-[0.06em] text-ink-muted">
      {title}
    </h4>
    {children}
  </div>
);

const Pill = ({ children }: { children: React.ReactNode }) => (
  <span className="inline-block rounded-full bg-primary-surface px-2 py-0.5 text-label font-medium text-primary-text">
    {children}
  </span>
);

function PresentScore({ preview }: { preview: PreviewMap }) {
  const p = preview as ScorePreview;
  const score = typeof p.aggregate_score === 'number' ? p.aggregate_score : null;
  const level = p.level ?? 'unknown';
  const dimensions = Array.isArray(p.dimensions) ? p.dimensions : [];
  const gap = p.gap_analysis ?? {};
  const num = (v: unknown, digits = 2) =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
  return (
    <div>
      <Section title="Score">
        <div className="text-2xl font-bold tabular-nums tracking-title text-ink">
          {score !== null ? score.toFixed(2) : '—'}{' '}
          <span className="ml-2 text-label font-medium uppercase tracking-[0.04em] text-ink-muted [overflow-wrap:anywhere]">{level}</span>
        </div>
      </Section>
      {dimensions.length > 0 && (
        <Section title="Dimensions">
          <ul className="space-y-1 text-sm">
            {dimensions.map((d, i) => (
              <li key={d?.name ?? i} className="flex justify-between gap-3">
                <span className="min-w-0 truncate">{d?.name ?? '—'}</span>
                <span className="shrink-0 tabular-nums text-ink-muted">
                  {num(d?.score)} × {Math.round((Number(d?.weight) || 0) * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </Section>
      )}
      <Section title="Gaps">
        <div className="text-sm tabular-nums">
          Matched: {(gap.matched_skills || []).length} · Transferable:{' '}
          {(gap.transferable_skills || []).length} · Missing:{' '}
          {(gap.missing_skills || []).length}
        </div>
      </Section>
      {p.interpretation && (
        <Section title="Interpretation">
          <p className="text-body text-ink">{p.interpretation}</p>
        </Section>
      )}
    </div>
  );
}

function ApproveSelection({ preview }: { preview: PreviewMap }) {
  const p = preview as SelectionPreview;
  const bullets = asArray<NonNullable<SelectionPreview['selected_bullets']>[number]>(p.selected_bullets);
  const order = asArray<string>(p.section_order);
  const emphasis = asArray<string>(p.sections_to_emphasize);
  return (
    <div>
      {order.length > 0 && (
        <Section title="Proposed section order">
          <div className="flex flex-wrap gap-1">
            {order.map((s, i) => (
              <Pill key={i}>{s}</Pill>
            ))}
          </div>
        </Section>
      )}
      {emphasis.length > 0 && (
        <Section title="Emphasised">
          <div className="flex flex-wrap gap-1">
            {emphasis.map((s, i) => (
              <Pill key={i}>{s}</Pill>
            ))}
          </div>
        </Section>
      )}
      {bullets.length > 0 && (
        <Section title={`Top bullets (${bullets.length})`}>
          <ul className="space-y-2">
            {bullets.slice(0, 8).map((b, i) => (
              <li
                key={i}
                className="rounded border border-border bg-surface-sunken p-2"
              >
                <div className="text-meta uppercase tracking-[0.06em] tabular-nums text-ink-muted">
                  {b?.section ?? '—'} · relevance{' '}
                  {typeof b?.relevance_score === 'number' ? b.relevance_score.toFixed(2) : '?'}
                </div>
                <div className="mt-0.5 text-body text-ink [overflow-wrap:anywhere]">{b?.original_text ?? ''}</div>
              </li>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
}

function ApproveRewrite({ preview }: { preview: PreviewMap }) {
  const p = preview as RewritePreview;
  const bullets = asArray<NonNullable<RewritePreview['rewritten_bullets']>[number]>(p.rewritten_bullets);
  const keywords = asArray<string>(p.keywords_inserted);
  return (
    <div>
      {keywords.length > 0 && (
        <Section title="Keywords woven in">
          <div className="flex flex-wrap gap-1">
            {keywords.map((k, i) => (
              <Pill key={i}>{k}</Pill>
            ))}
          </div>
        </Section>
      )}
      <Section title={`Rewrites (${bullets.length})`}>
        <ul className="space-y-2">
          {bullets.slice(0, 8).map((b, i) => (
            <li
              key={i}
              className="rounded border border-border bg-surface-sunken p-2"
            >
              <div className="text-meta uppercase tracking-[0.06em] tabular-nums text-ink-muted">
                confidence {typeof b?.confidence === 'number' ? b.confidence.toFixed(2) : '?'}
              </div>
              <div className="mt-1 rounded bg-removed-surface px-1 text-body text-removed-text line-through decoration-1 [overflow-wrap:anywhere]">
                {b?.original ?? ''}
              </div>
              <div className="mt-1 rounded bg-added-surface px-1 text-body text-added-text [overflow-wrap:anywhere]">
                {b?.rewritten ?? ''}
              </div>
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}

function ApproveCoverLetter({ preview }: { preview: PreviewMap }) {
  const p = preview as CoverLetterPreview;
  const paras = [
    p.opening_paragraph,
    p.body_paragraph_1,
    p.body_paragraph_2,
    p.body_paragraph_3,
    p.closing_paragraph,
  ].filter((s): s is string => typeof s === 'string' && s.trim().length > 0);
  return (
    <div>
      {p.betreff && (
        <Section title="Betreff">
          <div className="text-body font-semibold [overflow-wrap:anywhere]">{p.betreff}</div>
        </Section>
      )}
      {paras.length > 0 ? (
        <Section title="Letter">
          <div className="max-w-[62ch] space-y-2.5 text-body text-ink [overflow-wrap:anywhere]">
            {paras.map((para, i) => (
              <p key={i}>{para}</p>
            ))}
          </div>
        </Section>
      ) : (
        <p className="text-body text-ink-muted">Draft not available yet.</p>
      )}
      {p.grussformel && (
        <Section title="Grußformel">
          <div className="text-body [overflow-wrap:anywhere]">{p.grussformel}</div>
        </Section>
      )}
    </div>
  );
}

function GenericJson({ preview }: { preview: PreviewMap }) {
  if (!preview || typeof preview !== 'object' || Object.keys(preview).length === 0) {
    return <p className="text-sm text-ink-muted">No preview data for this gate.</p>;
  }
  let json = '';
  try {
    json = JSON.stringify(preview, null, 2);
  } catch {
    return <p className="text-sm text-ink-muted">This preview couldn’t be displayed.</p>;
  }
  return (
    <pre className="max-h-64 overflow-auto rounded bg-surface-sunken p-2 text-xs text-ink-muted">
      {json}
    </pre>
  );
}

export const PreviewCard = ({ gate }: PreviewCardProps) => {
  switch (gate.step) {
    case 'present_score':
      return <PresentScore preview={gate.preview} />;
    case 'approve_selection':
      return <ApproveSelection preview={gate.preview} />;
    case 'approve_rewrite':
      return <ApproveRewrite preview={gate.preview} />;
    case 'approve_cover_letter':
      return <ApproveCoverLetter preview={gate.preview} />;
    default:
      return <GenericJson preview={gate.preview} />;
  }
};
