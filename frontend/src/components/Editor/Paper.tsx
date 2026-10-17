/**
 * Paper - a page that scales with the pane, so a document laid out in points
 * (as the LaTeX templates are) keeps its proportions at any width.
 *
 * The sheet is a size container: `pt(n)` gives a length of n typographic
 * points on the printed page, as a share of the sheet's width (the page width
 * in points is the --page-width the sheet sets). The paper stays white with
 * black ink in both themes, like the PDF it stands for.
 */

import { useLayoutEffect, useRef } from 'react';

/** n points on the printed page, scaled to the sheet. */
export const pt = (n: number) => `calc(${n} * 100cqw / var(--page-width))`;

/** Page sizes in points (width, height). */
export const A4: [number, number] = [595.3, 841.9];
export const LETTER: [number, number] = [612, 792];

/** The LaTeX templates' Latin Modern, with close system serifs after it. */
const PAPER_FONT = '"Latin Modern Roman", "CMU Serif", "Computer Modern Serif", Cambria, "Times New Roman", serif';

/**
 * `page` is the paper size in points; `fontSize` the document class size in
 * points; `lineHeight` its baseline skip over that size; `margin` the page
 * margins in points (top, sides, bottom).
 */
export const Paper = ({
  children,
  label,
  page,
  fontSize,
  lineHeight,
  margin,
}: {
  children: React.ReactNode;
  label: string;
  page: [number, number];
  fontSize: number;
  lineHeight: number;
  margin: [number, number, number];
}) => (
  <div
    className="[container-type:inline-size] mx-auto w-full max-w-[52rem]"
    style={{ '--page-width': page[0] } as React.CSSProperties}
  >
    <article
      aria-label={label}
      className="relative bg-white text-black shadow-lg shadow-black/10 ring-1 ring-black/5"
      style={{
        minHeight: pt(page[1]),
        padding: `${pt(margin[0])} ${pt(margin[1])} ${pt(margin[2])}`,
        fontFamily: PAPER_FONT,
        fontSize: pt(fontSize),
        lineHeight,
      }}
    >
      {children}
    </article>
  </div>
);

const fitHeight = (el: HTMLTextAreaElement | null) => {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = `${el.scrollHeight}px`;
};

/**
 * A textarea that prints like the text around it and grows with its content,
 * so a paragraph being edited takes the space it takes on the page.
 */
export const PaperText = ({
  value,
  onChange,
  label,
  disabled,
  className = '',
  style,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
  placeholder?: string;
}) => {
  const ref = useRef<HTMLTextAreaElement>(null);
  // Fit the height to the text after every render, and again when the sheet's
  // width changes (the text wraps differently then).
  useLayoutEffect(() => fitHeight(ref.current));
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    // Only a width change: the height change fitHeight makes is observed too.
    let width = el.clientWidth;
    const observer = new ResizeObserver(() => {
      if (el.clientWidth === width) return;
      width = el.clientWidth;
      fitHeight(el);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <textarea
      ref={ref}
      rows={1}
      value={value}
      aria-label={label}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={`-mx-1 block w-[calc(100%+0.5rem)] resize-none overflow-hidden rounded-sm bg-transparent px-1 font-[inherit] leading-[inherit] text-inherit outline-none transition-colors placeholder:text-black/35 hover:bg-primary/[0.07] focus:bg-primary/[0.09] focus:ring-1 focus:ring-primary/60 disabled:hover:bg-transparent ${className}`}
      style={{ fontSize: 'inherit', ...style }}
    />
  );
};
