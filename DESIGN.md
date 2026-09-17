# Design

<!-- impeccable:design-schema 1 -->

Durable visual decisions for the JobWriterAI frontend redesign (`feature/redesign`).
Product truth lives in [PRODUCT.md](PRODUCT.md). This file records the visual
world; it describes what is built, not an aspiration.

## World

**"You see it before it's yours" — warm editorial.** The interface is paper and
ink with one terracotta voice. It refuses the black-box "tailor my resume"
button and the centered-hero-over-an-icon-card-grid entry: the landing proves the
human-in-the-loop mechanism by rendering a real compatibility gate — a fit
score, weighted factors, and Approve / Edit / Reject — in the first viewport.

Source of truth for the direction: user-authored Figma comp "Concept B — split
showcase" (`figma.com/design/0gOzPfmywb5W8tFJ11Z0BP`). No concept tournament —
the direction was pinned.

Modes: the landing/entry surface is **Persuade** (make the mechanism intelligible
and desirable, expose one action). The chat + gate workspace is **Operate**
(scanability and the task outrank expression) and inherits these tokens but not
this file's compositional choices.

## Color

Semantic token system in `frontend/src/app/globals.css`. Primitives are declared
in oklch on `:root` (light) and `.dark`; `@theme inline` exposes them to Tailwind
utilities (`bg-canvas`, `text-ink-muted`, `border-border`, `bg-primary`, …).
A theme switch remaps every role at once; `useTheme` / a pre-paint script toggle
`.dark` / `.light` on `<html>`.

Strategy: **restrained** — warm neutrals carry the surface, one terracotta accent
owns every call to action, link, selected state, and the emphasized half of the
hero headline. Light is the primary scene (a student at a laptop, daytime); dark
is a full peer.

| Role | Light | Use |
|---|---|---|
| `canvas` | `oklch(0.974 0.010 88)` — warm paper | page ground |
| `surface` | `oklch(0.995 0.003 88)` — near-white | cards, panels |
| `surface-sunken` | `oklch(0.901 0.017 82)` — sand | the flow-chooser band, insets |
| `ink` | `oklch(0.255 0.009 65)` — warm near-black | headings, primary text |
| `ink-muted` | `oklch(0.487 0.014 68)` | body, secondary text (AA on canvas) |
| `ink-faint` | `oklch(0.596 0.013 70)` | decorative / non-essential only |
| `border` | `oklch(0.893 0.014 84)` | hairlines |
| `border-strong` | `oklch(0.826 0.021 84)` | outline controls, dashed chips |
| `primary` | `oklch(0.585 0.128 40)` — terracotta | CTA fills, accent bars, featured border |
| `primary-hover` | `oklch(0.520 0.125 39)` | — |
| `primary-fg` | `oklch(0.968 0.010 60)` | text/icons on a terracotta fill |
| `primary-text` | `oklch(0.475 0.130 40)` | terracotta text/links on paper (AA) |
| `primary-surface` | `oklch(0.918 0.030 48)` | terracotta tint (icon chips) |
| `ring` | `= primary` | focus ring |

Dark mode: warm brown-black grounds (`canvas oklch(0.178 0.008 60)`), lifted
terracotta (`primary oklch(0.660 0.128 44)`, `primary-text oklch(0.780 0.095 46)`).
`success` / `danger` / `warning` / diff (`added` / `removed`) roles are unchanged
from the prior system and remain semantic-only.

Browser surfaces are themed from the palette: text selection (`primary` at 26%),
`:focus-visible` (2px `ring`, 2px offset), scrollbars (`border-strong` thumb).

## Typography

Fonts via `next/font/google` in `frontend/src/app/layout.tsx`, exposed as CSS
variables and mapped in `@theme`:

- **Display + wordmark — Libre Caslon Text** (`--font-serif-user`, `font-serif`).
  Weights 400 / 700 + italic. Headings use `font-normal`; do **not** request 600
  (unshipped → faux-bold fallback). Hero display runs to `text-6xl`.
- **Body + UI — Libre Franklin** (`--font-sans-user`, the default `font-sans` and
  `body`). Variable weight.
- **Mono — Geist Mono** (`--font-geist-mono`), for data / measurement only.

Named size roles (set in `@theme`, beyond Tailwind's numeric scale):
`meta` .75rem · `label` .8125rem · `sm` .875rem · `body` .9375rem (reading floor
for prose, chat, gate narration) · `lede` 1.125rem (hero subhead, section intros).
Display/title sizes stay on `text-2xl`…`6xl` with `tracking-display` (-0.021em) /
`tracking-title` (-0.014em); the hero headline uses `tracking-[-0.005em]`.

Heading case is **sentence case** everywhere ("Or choose where to start",
"How it works", "Your resume").

## Layout & materials

- Content max width `max-w-6xl` (hero) / `max-w-5xl` (chooser, how-it-works) /
  `max-w-4xl` (upload), centered, `px-6` gutter.
- **Hero**: `lg:grid-cols-[1.05fr_0.95fr]`, `lg:items-start` (columns top-aligned —
  the tall gate card must not vertically center the short text column). Left:
  headline → `lede` subcopy → one pill CTA. Right: the gate demo card.
- Corners: cards `rounded-2xl` (~16px); the gate demo and chooser inner panel
  `rounded-[18px]`; the chooser outer band `rounded-[24px]`; buttons and chips are
  **full-round** (`rounded-full`).
- Elevation: one soft, warm, offset shadow —
  `shadow-[0_22px_50px_-18px_rgba(60,38,22,0.28)]` on the gate card,
  lighter (`shadow-lg shadow-black/5`) on content cards. No zero-blur / hard
  shadows. The gate demo card and its back-peek card carry a slight rotation
  (`rotate(-1.4deg)` / `rotate(3deg)`) — this is the only rotated element.
- Borders are 1px hairlines; the featured chooser card is the one exception
  (`border-2 border-primary`), lifted with `sm:-my-3`.
- Icons: drawn line SVGs, 1.5–1.75 stroke, in `h-9 w-9` circular tinted chips.
  No icon fonts, no emoji.
- Unavailable affordances (Discovery flow): dashed `border-strong` chip, muted
  copy, "Coming soon", `cursor-not-allowed`, `aria-disabled` — discoverable,
  clearly not selectable.

## Motion

One authored entrance on the landing: `heroRise` (translateY 16px + 6px blur →
rest, `cubic-bezier(0.16, 1, 0.3, 1)`), text column first (~0.7s), gate card
`0.14s` behind it (`.hero-rise` / `.hero-rise-delayed`). CTAs and the featured
card lift on hover (`-translate-y-0.5`, 200ms). Everything above is disabled
under `prefers-reduced-motion`; color transitions and the `animate-spin` wait
indicator are kept.

## Components

New, under `frontend/src/components/landing/`:

- **`LandingHero`** — the split hero. `onStart` runs the default `job_tailoring`
  flow.
- **`GateDemo`** — static, non-interactive preview of a compatibility gate
  (score `0.78`, weighted factor bars, Approve/Edit/Reject as styled divs, the
  "nothing written until you say so" chip). One `role="img"` label on the wrapper;
  inner controls are `aria-hidden` — no keyboard trap.
- **`FlowChooser`** — the sand band. Three named flows in a connected panel;
  "Tailor to a job" featured; "Discover careers" disabled.

Reworked: `Header` (serif wordmark + "Pre-launch" tag + theme toggle),
`UploadSection` (landing branch = `LandingHero` + `FlowChooser` + `HowItWorks`;
chosen-flow branch keeps the upload cards, restyled), `HowItWorks` (serif
sentence-case heading). `page.tsx` hides the progress breadcrumb until a flow is
chosen.

## Accessibility

Target **WCAG 2.1 AA** (see PRODUCT.md). Body/secondary text uses `ink-muted`,
which clears 4.5:1 on `canvas` and `surface`; `ink-faint` is reserved for
non-essential text. Terracotta-on-paper text uses `primary-text`, not `primary`.
The gate diff/score is conveyed by number + label, never color alone. Focus is
visible on every interactive element (2px `ring`, 2px offset). Long generation
waits must be announced to assistive tech, not shown only as spinners.

## Not yet in this world

The chat container, gate panels (`ApprovalGate`, `ChoiceGate`, `InputGate`),
`PreviewCard`, upload/job-input internals, and the analysis loading screen use
the new tokens but have not had their composition redesigned to Concept B. That
is the next tranche of the redesign.
