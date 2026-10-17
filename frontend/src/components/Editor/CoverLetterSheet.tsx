/**
 * CoverLetterSheet - the cover letter laid out as its PDF prints it
 * (backend/app/templates/latex/cover_letter*.tex.j2), with the parts the user
 * may change edited in place: the Betreff line, the paragraphs and the
 * sign-off. The header comes from the CV and the salutation from the
 * recipient, so they are shown, not edited, here.
 */

import { useState } from 'react';
import type { CoverLetterPrint } from '@/types';
import { Paper, PaperText, pt } from './Paper';

/** The templates' UI_blue. */
const BLUE = 'rgb(32, 64, 151)';
/** The templates' footer grey. */
const GREY = 'rgb(179, 179, 179)';
/** The justified part of the letter indents each paragraph's first line (ragged2e's
 * \JustifyingParindent, 17pt in an 11pt document); the sign-off block doesn't. */
const INDENT = pt(17);

const PARAGRAPHS: { field: string; label: string }[] = [
  { field: 'opening_paragraph', label: 'Opening paragraph' },
  { field: 'body_paragraph_1', label: 'First body paragraph' },
  { field: 'body_paragraph_2', label: 'Second body paragraph' },
  { field: 'body_paragraph_3', label: 'Third body paragraph' },
  { field: 'closing_paragraph', label: 'Closing paragraph' },
];

interface CoverLetterSheetProps {
  /** What the PDF prints (header, date, salutation), from the server's view. */
  letter: CoverLetterPrint;
  /** The editable values being edited. */
  fields: Record<string, string>;
  onChange: (field: string, value: string) => void;
  disabled: boolean;
}

export const CoverLetterSheet = ({ letter, fields, onChange, disabled }: CoverLetterSheetProps) => {
  const german = 'betreff' in fields;
  // The optional third body paragraph only takes space once it has text, as in
  // the PDF, or once the user asked for it.
  const [addingThird, setAddingThird] = useState(false);
  const showThird = addingThird || Boolean(fields.body_paragraph_3?.trim());
  const paragraphs = PARAGRAPHS.filter((p) => p.field !== 'body_paragraph_3' || showThird);
  const links = [letter.link_1_label, letter.link_2_label].filter(Boolean);
  const fromCv = 'From your CV; change it there.';

  return (
    // 11pt article, 1cm margins (less at the top, where CSS adds half-leading
    // above the first line and LaTeX doesn't); LaTeX's 13.6pt baseline skip.
    <Paper label="Cover letter" fontSize={11} lineHeight={13.6 / 11} margin={[22, 28.35, 28.35]}>
      <header
        className="grid items-end"
        style={{ gridTemplateColumns: '24% 50% 24%', columnGap: '1%' }}
        title={fromCv}
      >
        <div style={{ fontSize: pt(12) }} className="[overflow-wrap:anywhere]">
          {letter.phone && <div>{letter.phone}</div>}
          {letter.email && <div>{letter.email}</div>}
        </div>
        <div className="text-center">
          <div style={{ fontSize: pt(24.88), lineHeight: 1.15 }}>{letter.name}</div>
          {letter.target_job && <div style={{ fontSize: pt(14.4), color: BLUE }}>{letter.target_job}</div>}
        </div>
        <div className="text-right [overflow-wrap:anywhere]" style={{ fontSize: pt(12) }}>
          {links.map((label) => (
            <div key={label}>{label}</div>
          ))}
        </div>
      </header>
      <hr style={{ border: 0, borderTop: `${pt(0.4)} solid ${BLUE}`, marginTop: pt(2) }} />

      <h2 className="text-center font-normal" style={{ fontSize: pt(14.4), color: BLUE, marginTop: pt(40) }}>
        {letter.document_title}
      </h2>

      <p style={{ marginTop: pt(23), textIndent: INDENT }}>
        {letter.date_label} {letter.current_date}
      </p>
      {german && (
        <div className="flex items-baseline" style={{ marginTop: pt(17.7), paddingLeft: INDENT }}>
          <strong className="shrink-0" style={{ marginRight: '0.33em' }}>
            Betreff:
          </strong>
          <PaperText
            value={fields.betreff ?? ''}
            onChange={(v) => onChange('betreff', v)}
            label="Betreff"
            disabled={disabled}
            placeholder="Bewerbung als …"
          />
        </div>
      )}
      <p style={{ marginTop: pt(9.2), textIndent: INDENT }}>{letter.salutation}</p>

      {paragraphs.map((p) => (
        <PaperText
          key={p.field}
          value={fields[p.field] ?? ''}
          onChange={(v) => onChange(p.field, v)}
          label={p.label}
          disabled={disabled}
          className="text-justify"
          style={{ marginTop: pt(12), textIndent: INDENT }}
        />
      ))}
      {!showThird && !disabled && (
        <button
          type="button"
          onClick={() => setAddingThird(true)}
          className="mt-2 rounded-sm px-1 text-label text-black/45 hover:bg-primary/[0.07] hover:text-black/70 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/60"
          style={{ fontFamily: 'var(--font-sans-user), system-ui, sans-serif' }}
        >
          + Add a third body paragraph
        </button>
      )}

      <div style={{ marginTop: pt(26.2) }}>
        <PaperText
          value={fields.sign_off ?? ''}
          onChange={(v) => onChange('sign_off', v)}
          label="Sign-off"
          disabled={disabled}
        />
        <div title={fromCv}>
          <div>{letter.name}</div>
          {letter.phone && <div>{letter.phone}</div>}
          {letter.email && <div>{letter.email}</div>}
        </div>
      </div>

      <footer className="absolute" style={{ right: pt(28.35), bottom: pt(28.35), color: GREY }}>
        {german ? `1 of 1, Aktualisiert am ${letter.current_date}` : `1 of 1, Updated ${letter.current_date}`}
      </footer>
    </Paper>
  );
};
