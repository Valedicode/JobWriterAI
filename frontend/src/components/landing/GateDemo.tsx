/**
 * A static, non-interactive preview of a compatibility gate — the signature
 * move of the landing hero. It shows the real mechanism (score → weighted
 * factors → approve / edit / reject) without wiring to the orchestrator.
 *
 * Everything inside is presentational: the "buttons" are divs, and the whole
 * card carries a single role="img" label so assistive tech gets the gist
 * without a keyboard trap.
 */

const factors = [
  { label: 'Hard skills', score: '0.82', weight: '40%', fill: 82 },
  { label: 'Experience', score: '0.71', weight: '35%', fill: 71 },
  { label: 'Keywords', score: '0.80', weight: '25%', fill: 80 },
];

export const GateDemo = () => {
  return (
    <div
      role="img"
      aria-label="Preview of a compatibility gate: fit score 0.78, a strong match, weighted across hard skills, experience, and keywords, with Approve, Edit, and Reject actions."
      className="relative"
    >
      {/* Card peeking behind, to suggest a stack of pending steps. */}
      <div
        aria-hidden="true"
        className="absolute -right-3 -top-3 h-full w-full rounded-[18px] border border-border bg-surface/60 [transform:rotate(3deg)]"
      />

      <div className="relative rounded-[18px] border border-border bg-surface p-6 shadow-[0_22px_50px_-18px_rgba(60,38,22,0.28)] [transform:rotate(-1.4deg)]">
        <p className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-primary-text">
          Compatibility gate
        </p>

        <div className="mt-3 flex items-baseline gap-3">
          <span className="font-serif text-5xl leading-none text-ink tabular-nums">
            0.78
          </span>
          <span className="text-sm text-ink-muted">Strong match</span>
        </div>

        <div className="mt-5 space-y-3.5 border-t border-border pt-5">
          {factors.map((f, i) => (
            <div key={f.label}>
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-ink">{f.label}</span>
                <span className="text-meta tabular-nums text-ink-faint">
                  {f.score} · {f.weight}
                </span>
              </div>
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-sunken">
                <div
                  className="animate-grow-x h-full rounded-full bg-primary"
                  style={{ width: `${f.fill}%`, animationDelay: `${0.35 + i * 0.09}s` }}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex flex-wrap gap-2 text-sm" aria-hidden="true">
          <span className="rounded-full bg-primary px-4 py-1.5 font-medium text-primary-fg">
            Approve
          </span>
          <span className="rounded-full border border-border-strong px-4 py-1.5 text-ink">
            Edit
          </span>
          <span className="rounded-full border border-border-strong px-4 py-1.5 text-ink">
            Reject
          </span>
        </div>
      </div>

      {/* Reassurance chip, overlapping the lower-left corner. */}
      <div className="absolute -bottom-4 left-4 inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1.5 text-meta text-ink-muted shadow-sm">
        <svg
          className="h-3.5 w-3.5 text-primary-text"
          viewBox="0 0 20 20"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M4 10.5l3.5 3.5L16 5.5"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        nothing written until you say so
      </div>
    </div>
  );
};
