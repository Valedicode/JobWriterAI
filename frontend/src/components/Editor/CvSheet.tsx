/**
 * CvSheet - the CV laid out as its PDF prints it (backend/app/templates/latex/
 * cv.tex.j2, the "normal" layout), with every piece of text that came from a
 * CV field edited in place. Text the template adds itself (separators,
 * "Technologies:", skill groups made from a plain list) is shown, not edited.
 * Bullet lists can grow and shrink.
 *
 * The text is edited in the page (contentEditable), not re-rendered while the
 * user types: an edit is reported when the field loses focus.
 */

import { useRef } from 'react';
import type { CvDocumentView, CvEntry, CvSegment, CvSection } from '@/types';
import { LETTER, Paper, pt } from './Paper';

interface CvSheetProps {
  view: CvDocumentView;
  disabled: boolean;
  /** A field's new text. */
  onSet: (path: string, value: string) => void;
  /** A new bullet at `index` of the list at `listPath`. */
  onInsert: (listPath: string, index: number) => void;
  /** Remove the bullet at `path`. */
  onDelete: (path: string) => void;
}

export const CvSheet = ({ view, disabled, onSet, onInsert, onDelete }: CvSheetProps) => {
  const { header, sections } = view;
  const segs = (list: CvSegment[], block = false) => (
    <Segments segments={list} disabled={disabled} onSet={onSet} block={block} />
  );
  return (
    // 10pt article on US Letter (the template sets no paper size), normal
    // layout: 0.96 linespread of the 12pt baseline skip, margins 0.8cm top,
    // 1.06cm sides, 0.6cm bottom (less at the top, where CSS adds half-leading
    // above the first line). The gaps below are measured on the PDF.
    <Paper label="CV" page={LETTER} fontSize={10} lineHeight={(12 * 0.96) / 10} margin={[18, 30.05, 17]}>
      <header className="text-center">
        <div className="font-bold" style={{ fontSize: pt(14.4) }}>
          {segs(header.name_segs)}
        </div>
        <hr style={{ border: 0, borderTop: `${pt(0.4)} solid black`, marginTop: pt(7) }} />
        {header.contact_segs.length > 0 && (
          <p style={{ marginTop: pt(12) }}>
            {header.contact_segs.map((part, i) => (
              <span key={i}>
                {'• '}
                {segs(part)}{' '}
              </span>
            ))}
          </p>
        )}
        {header.label_segs.length > 0 && <p style={{ marginTop: pt(12) }}>{segs(header.label_segs)}</p>}
        {header.summary_segs.length > 0 && (
          <p style={{ marginTop: pt(20), marginBottom: pt(7) }}>{segs(header.summary_segs, true)}</p>
        )}
      </header>

      {sections.map((section) => (
        <Section key={section.type} section={section} segs={segs} disabled={disabled} onInsert={onInsert} onDelete={onDelete} />
      ))}
    </Paper>
  );
};

const Section = ({
  section,
  segs,
  disabled,
  onInsert,
  onDelete,
}: {
  section: CvSection;
  segs: (list: CvSegment[], block?: boolean) => React.ReactNode;
  disabled: boolean;
  onInsert: CvSheetProps['onInsert'];
  onDelete: CvSheetProps['onDelete'];
}) => (
  <section style={{ marginTop: pt(10) }}>
    <h3 className="text-center font-bold" style={{ fontSize: pt(12), marginBottom: pt(9) }}>
      {section.title}
    </h3>
    {section.style === 'itemize' ? (
      <ul className="list-disc" style={{ paddingLeft: '1.3em' }}>
        {section.entries.map((entry, i) => (
          <li key={i}>
            {entry.label && (
              <strong>
                {segs(entry.label_segs ?? [])}
                {':'}
              </strong>
            )}{' '}
            {segs(entry.text_segs ?? [])}
          </li>
        ))}
      </ul>
    ) : (
      section.entries.map((entry, i) => (
        <Entry
          key={i}
          entry={entry}
          first={i === 0}
          segs={segs}
          disabled={disabled}
          onInsert={onInsert}
          onDelete={onDelete}
        />
      ))
    )}
  </section>
);

const Entry = ({
  entry,
  first,
  segs,
  disabled,
  onInsert,
  onDelete,
}: {
  entry: CvEntry;
  first: boolean;
  segs: (list: CvSegment[], block?: boolean) => React.ReactNode;
  disabled: boolean;
  onInsert: CvSheetProps['onInsert'];
  onDelete: CvSheetProps['onDelete'];
}) => {
  const listPath = entry.bullets_src?.[0];
  const bullets = entry.bullet_segs ?? [];
  // A bullet's own path ends in its index in the CV's list (empty bullets are skipped when printed).
  const indexOf = (bullet: CvSegment[]) => Number(bullet[0]?.src[0]?.split('.').pop());
  return (
    <div className="group/entry relative" style={{ marginTop: first ? 0 : pt(4) }}>
      {(entry.rows ?? []).map((row, i) => (
        <div key={i} className="flex justify-between" style={{ gap: pt(8) }}>
          <span className={row.bold ? 'font-bold' : undefined}>{segs(row.left_segs)}</span>
          {row.right_segs.length > 0 && <span className="shrink-0 text-right">{segs(row.right_segs)}</span>}
        </div>
      ))}
      {bullets.length > 0 && (
        <ul className="list-disc" style={{ paddingLeft: '1.3em', marginTop: pt(1) }}>
          {bullets.map((bullet, i) => (
            <li key={`${bullet[0]?.src[0] ?? i}`} className="group/bullet relative">
              {segs(bullet, true)}
              {!disabled && bullet[0]?.src[0] && (
                <span className={controls} style={{ right: '-0.4em' }}>
                  <ControlButton label="Add a bullet below" onClick={() => listPath && onInsert(listPath, indexOf(bullet) + 1)}>
                    +
                  </ControlButton>
                  <ControlButton label="Remove this bullet" onClick={() => onDelete(bullet[0].src[0])}>
                    ×
                  </ControlButton>
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {!disabled && listPath && bullets.length === 0 && (
        <button type="button" onClick={() => onInsert(listPath, 0)} className={addBullet} style={uiFont}>
          + Add a bullet
        </button>
      )}
    </div>
  );
};

/** One run of printed text: editable pieces for CV fields, plain text for the template's own words. */
const Segments = ({
  segments,
  disabled,
  onSet,
  block,
}: {
  segments: CvSegment[];
  disabled: boolean;
  onSet: (path: string, value: string) => void;
  block: boolean;
}) => (
  <>
    {segments.map((seg, i) =>
      seg.src.length > 0 && !disabled ? (
        <EditableText key={i} seg={seg} onSet={onSet} block={block} />
      ) : (
        <span key={i} title={seg.src.length === 0 && seg.text.trim().length > 2 ? fixedHint : undefined}>
          {seg.text}
        </span>
      )
    )}
  </>
);

const EditableText = ({
  seg,
  onSet,
  block,
}: {
  seg: CvSegment;
  onSet: (path: string, value: string) => void;
  block: boolean;
}) => {
  // The text this field was last reported with, so an edit is reported once.
  const reported = useRef(seg.text);
  return (
    <span
      contentEditable="plaintext-only"
      suppressContentEditableWarning
      role="textbox"
      aria-label={seg.kind === 'list' ? `${fieldName(seg.src[0])} (comma-separated)` : fieldName(seg.src[0])}
      aria-multiline={block}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          e.currentTarget.blur();
        }
      }}
      onBlur={(e) => {
        const text = (e.currentTarget.textContent ?? '').replace(/\s+/g, ' ').trim();
        if (text === reported.current.trim()) return;
        reported.current = text;
        onSet(seg.src[0], text);
      }}
      className="rounded-sm outline-none transition-colors hover:bg-primary/[0.08] focus:bg-primary/[0.1] focus:ring-1 focus:ring-primary/60"
    >
      {seg.text}
    </span>
  );
};

const ControlButton = ({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) => (
  <button
    type="button"
    aria-label={label}
    title={label}
    onClick={onClick}
    className="flex h-5 w-5 items-center justify-center rounded-full bg-white text-sm leading-none text-black/60 shadow ring-1 ring-black/10 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
    style={uiFont}
  >
    {children}
  </button>
);

/** "experience.0.company" -> "experience 1 company", for screen readers. */
const fieldName = (path: string) =>
  path.replace(/\.(\d+)/g, (_, n: string) => ` ${Number(n) + 1}`).replace(/[._]/g, ' ').trim();

const fixedHint = "Added by the template; it isn't a field of your CV.";
const uiFont: React.CSSProperties = { fontFamily: 'var(--font-sans-user), system-ui, sans-serif' };
const controls =
  'absolute top-0 flex translate-x-full gap-1 pl-1 opacity-0 transition-opacity group-hover/bullet:opacity-100 group-focus-within/bullet:opacity-100';
// Below the entry without taking space, so the page keeps the PDF's spacing.
const addBullet =
  'absolute left-0 top-full z-10 rounded-sm bg-white px-1 text-label text-black/45 opacity-0 transition-opacity hover:bg-primary/[0.07] hover:text-black/70 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/60 group-hover/entry:opacity-100';
