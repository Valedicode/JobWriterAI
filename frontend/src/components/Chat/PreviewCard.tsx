/**
 * PreviewCard - typed renderer for the preview payload of an orchestrator gate.
 *
 * Job-tailoring steps have bespoke renderers below. Everything else — the CV
 * review sections, plus any future step — goes through StructuredPreview, which
 * turns the payload object into readable sections, lists, and diffs. Nothing is
 * ever dumped as raw JSON.
 */

import type { OrchestratorGatePayload } from '@/types';
import { humanizeKey } from './gateFormat';

type PreviewMap = Record<string, unknown>;

interface PreviewCardProps {
  gate: OrchestratorGatePayload;
}

/** Coerce a possibly-malformed backend field to a safe array. */
const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);

const num = (v: unknown, digits = 2) =>
  typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';

// --------------------------------------------------------------------------
// Shared bits
// --------------------------------------------------------------------------

const Section = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <div className="mb-4 last:mb-0">
    <h4 className="mb-1.5 text-meta font-semibold uppercase tracking-[0.12em] text-ink-muted">
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

const VerdictBadge = ({ value }: { value: string }) => {
  const v = value.toLowerCase();
  const tone =
    v.includes('strong') || v.includes('excellent') || v.includes('good')
      ? 'bg-success-surface text-success-text border-success-border'
      : v.includes('weak') || v.includes('poor') || v.includes('needs')
        ? 'bg-warning-surface text-warning-text border-warning-border'
        : 'bg-surface-sunken text-ink-muted border-border';
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-label font-medium ${tone}`}
    >
      {value}
    </span>
  );
};

const Bullets = ({ items }: { items: string[] }) => (
  <ul className="space-y-1.5 text-body text-ink">
    {items.map((s, i) => (
      <li key={i} className="flex gap-2 [overflow-wrap:anywhere]">
        <span aria-hidden className="mt-[0.55em] h-1 w-1 flex-shrink-0 rounded-full bg-primary" />
        <span>{s}</span>
      </li>
    ))}
  </ul>
);

const Diff = ({ original, rewritten }: { original?: string; rewritten?: string }) => (
  <div className="space-y-1">
    {original ? (
      <div className="rounded bg-removed-surface px-1.5 py-0.5 text-body text-removed-text line-through decoration-1 [overflow-wrap:anywhere]">
        {original}
      </div>
    ) : null}
    {rewritten ? (
      <div className="rounded bg-added-surface px-1.5 py-0.5 text-body text-added-text [overflow-wrap:anywhere]">
        {rewritten}
      </div>
    ) : null}
  </div>
);

const MiniCard = ({ children }: { children: React.ReactNode }) => (
  <li className="rounded-lg border border-border bg-surface-sunken/60 p-2.5">{children}</li>
);

// --------------------------------------------------------------------------
// Job-tailoring step shapes (bespoke)
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

function PresentScore({ preview }: { preview: PreviewMap }) {
  const p = preview as ScorePreview;
  const score = typeof p.aggregate_score === 'number' ? p.aggregate_score : null;
  const level = p.level ?? 'unknown';
  const dimensions = Array.isArray(p.dimensions) ? p.dimensions : [];
  const gap = p.gap_analysis ?? {};
  return (
    <div>
      <Section title="Score">
        <div className="flex items-baseline gap-3">
          <span className="font-serif text-4xl leading-none tabular-nums text-ink">
            {score !== null ? score.toFixed(2) : '—'}
          </span>
          <span className="text-sm capitalize text-ink-muted [overflow-wrap:anywhere]">
            {level}
          </span>
        </div>
      </Section>
      {dimensions.length > 0 && (
        <Section title="Dimensions">
          <ul className="space-y-2.5">
            {dimensions.map((d, i) => {
              const s = Number(d?.score) || 0;
              const w = Math.round((Number(d?.weight) || 0) * 100);
              return (
                <li key={d?.name ?? i}>
                  <div className="flex items-baseline justify-between text-sm">
                    <span className="min-w-0 truncate text-ink">{d?.name ?? '—'}</span>
                    <span className="shrink-0 tabular-nums text-ink-faint">
                      {num(d?.score)} · {w}%
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-sunken">
                    <div
                      className="animate-grow-x h-full rounded-full bg-primary"
                      style={{
                        width: `${Math.max(0, Math.min(1, s)) * 100}%`,
                        animationDelay: `${i * 70}ms`,
                      }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </Section>
      )}
      <Section title="Skill gaps">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm tabular-nums text-ink-muted">
          <span>Matched {(gap.matched_skills || []).length}</span>
          <span>Transferable {(gap.transferable_skills || []).length}</span>
          <span>Missing {(gap.missing_skills || []).length}</span>
        </div>
      </Section>
      {p.interpretation && (
        <Section title="Interpretation">
          <p className="text-body text-ink [overflow-wrap:anywhere]">{p.interpretation}</p>
        </Section>
      )}
    </div>
  );
}

interface SelectionPreview {
  selected_bullets?: Array<{ section: string; original_text: string; relevance_score: number }>;
  section_order?: string[];
  sections_to_emphasize?: string[];
}

function ApproveSelection({ preview }: { preview: PreviewMap }) {
  const p = preview as SelectionPreview;
  const bullets = asArray<NonNullable<SelectionPreview['selected_bullets']>[number]>(
    p.selected_bullets,
  );
  const order = asArray<string>(p.section_order);
  const emphasis = asArray<string>(p.sections_to_emphasize);
  return (
    <div>
      {order.length > 0 && (
        <Section title="Proposed section order">
          <div className="flex flex-wrap gap-1.5">
            {order.map((s, i) => (
              <Pill key={i}>{s}</Pill>
            ))}
          </div>
        </Section>
      )}
      {emphasis.length > 0 && (
        <Section title="Emphasised">
          <div className="flex flex-wrap gap-1.5">
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
              <MiniCard key={i}>
                <div className="text-meta uppercase tracking-[0.08em] tabular-nums text-ink-faint">
                  {b?.section ?? '—'} · relevance{' '}
                  {typeof b?.relevance_score === 'number' ? b.relevance_score.toFixed(2) : '?'}
                </div>
                <div className="mt-1 text-body text-ink [overflow-wrap:anywhere]">
                  {b?.original_text ?? ''}
                </div>
              </MiniCard>
            ))}
          </ul>
        </Section>
      )}
    </div>
  );
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

function ApproveRewrite({ preview }: { preview: PreviewMap }) {
  const p = preview as RewritePreview;
  const bullets = asArray<NonNullable<RewritePreview['rewritten_bullets']>[number]>(
    p.rewritten_bullets,
  );
  const keywords = asArray<string>(p.keywords_inserted);
  return (
    <div>
      {keywords.length > 0 && (
        <Section title="Keywords woven in">
          <div className="flex flex-wrap gap-1.5">
            {keywords.map((k, i) => (
              <Pill key={i}>{k}</Pill>
            ))}
          </div>
        </Section>
      )}
      <Section title={`Rewrites (${bullets.length})`}>
        <ul className="space-y-2">
          {bullets.slice(0, 8).map((b, i) => (
            <MiniCard key={i}>
              <div className="text-meta uppercase tracking-[0.08em] tabular-nums text-ink-faint">
                confidence {typeof b?.confidence === 'number' ? b.confidence.toFixed(2) : '?'}
              </div>
              <div className="mt-1.5">
                <Diff original={b?.original} rewritten={b?.rewritten} />
              </div>
            </MiniCard>
          ))}
        </ul>
      </Section>
    </div>
  );
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
          <div className="text-body font-medium text-ink [overflow-wrap:anywhere]">{p.betreff}</div>
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
          <div className="text-body text-ink [overflow-wrap:anywhere]">{p.grussformel}</div>
        </Section>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------
// Structured fallback — handles CV review sections and anything unrecognised
// --------------------------------------------------------------------------

const isNonEmptyString = (v: unknown): v is string =>
  typeof v === 'string' && v.trim().length > 0;
const isStringArray = (v: unknown): v is string[] =>
  Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === 'string');
const isObjectArray = (v: unknown): v is Record<string, unknown>[] =>
  Array.isArray(v) &&
  v.length > 0 &&
  v.every((x) => x !== null && typeof x === 'object' && !Array.isArray(x));
const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const hasKeys = (o: Record<string, unknown>, ...keys: string[]) =>
  keys.every((k) => k in o);

// Keys shown first, in this order, when present.
const KEY_PRIORITY = [
  'verdict',
  'score',
  'strengths',
  'critique',
  'weaknesses',
  'role_critiques',
  'issues',
  'suggestions',
  'top_improvements',
  'improved_entries',
  'rewritten_header',
  'rewritten_bullets',
  'refined_project_descriptions',
];

function orderedEntries(preview: PreviewMap): [string, unknown][] {
  const entries = Object.entries(preview).filter(([, v]) => {
    if (v === null || v === undefined || v === false) return false;
    if (typeof v === 'string') return v.trim().length > 0;
    if (Array.isArray(v)) return v.length > 0;
    if (typeof v === 'object') return Object.keys(v).length > 0;
    return true;
  });
  return entries.sort(([a], [b]) => {
    const ia = KEY_PRIORITY.indexOf(a);
    const ib = KEY_PRIORITY.indexOf(b);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });
}

function renderObjectArray(key: string, items: Record<string, unknown>[]) {
  return (
    <Section key={key} title={humanizeKey(key)}>
      <ul className="space-y-2">
        {items.slice(0, 10).map((item, i) => {
          if (hasKeys(item, 'original', 'rewritten')) {
            return (
              <MiniCard key={i}>
                {isNonEmptyString(item.rationale) && (
                  <div className="mb-1.5 text-meta text-ink-faint [overflow-wrap:anywhere]">
                    {item.rationale}
                  </div>
                )}
                <Diff
                  original={String(item.original ?? '')}
                  rewritten={String(item.rewritten ?? '')}
                />
              </MiniCard>
            );
          }
          if (hasKeys(item, 'name') && isStringArray(item.skills)) {
            return (
              <MiniCard key={i}>
                <div className="mb-1 text-sm font-medium text-ink">{String(item.name)}</div>
                <div className="flex flex-wrap gap-1.5">
                  {item.skills.map((s, j) => (
                    <Pill key={j}>{s}</Pill>
                  ))}
                </div>
              </MiniCard>
            );
          }
          if (hasKeys(item, 'project', 'description')) {
            return (
              <MiniCard key={i}>
                <div className="text-sm font-medium text-ink">{String(item.project)}</div>
                <div className="mt-0.5 text-body text-ink [overflow-wrap:anywhere]">
                  {String(item.description)}
                </div>
              </MiniCard>
            );
          }
          return (
            <MiniCard key={i}>
              <StructuredBody preview={item} dense />
            </MiniCard>
          );
        })}
      </ul>
    </Section>
  );
}

function StructuredBody({ preview, dense }: { preview: PreviewMap; dense?: boolean }) {
  const entries = orderedEntries(preview);
  if (entries.length === 0) {
    return <p className="text-body text-ink-muted">No details for this step.</p>;
  }

  return (
    <div className={dense ? 'space-y-2' : ''}>
      {entries.map(([key, value]) => {
        if (key === 'verdict' && isNonEmptyString(value)) {
          return (
            <Section key={key} title="Verdict">
              <VerdictBadge value={value} />
            </Section>
          );
        }
        if (key === 'score' && typeof value === 'number') {
          return (
            <Section key={key} title="Score">
              <span className="font-serif text-3xl tabular-nums text-ink">
                {value % 1 === 0 ? value : value.toFixed(1)}
                {value <= 10 ? <span className="text-lg text-ink-faint"> / 10</span> : null}
              </span>
            </Section>
          );
        }
        if (typeof value === 'number') {
          return (
            <Section key={key} title={humanizeKey(key)}>
              <span className="tabular-nums text-body text-ink">{value}</span>
            </Section>
          );
        }
        if (isNonEmptyString(value)) {
          return (
            <Section key={key} title={humanizeKey(key)}>
              <p className="text-body text-ink [overflow-wrap:anywhere]">{value}</p>
            </Section>
          );
        }
        if (isStringArray(value)) {
          return (
            <Section key={key} title={humanizeKey(key)}>
              <Bullets items={value.slice(0, 12)} />
            </Section>
          );
        }
        if (isObjectArray(value)) {
          return renderObjectArray(key, value);
        }
        if (isPlainObject(value)) {
          return (
            <Section key={key} title={humanizeKey(key)}>
              <div className="rounded-lg border border-border bg-surface-sunken/50 p-2.5">
                <StructuredBody preview={value} dense />
              </div>
            </Section>
          );
        }
        return null;
      })}
    </div>
  );
}

function StructuredPreview({ preview }: { preview: PreviewMap }) {
  if (!isPlainObject(preview) || Object.keys(preview).length === 0) {
    return <p className="text-body text-ink-muted">No preview for this step.</p>;
  }
  return <StructuredBody preview={preview} />;
}

// --------------------------------------------------------------------------

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
      return <StructuredPreview preview={gate.preview} />;
  }
};
