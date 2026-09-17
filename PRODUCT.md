# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are **students and recent graduates** applying to jobs, typically
early in their careers and applying to many roles. They arrive with a single
generic PDF resume and a specific job posting (URL or pasted text), and they
want application materials that are tailored to that posting without having to
rewrite everything by hand. They are not expert resume writers and benefit from
seeing what the system changed and why before it commits.

## Product Purpose

JobWriterAI turns a generic resume plus a target job description into a
tailored resume and a matching cover letter, produced by a multi-agent pipeline
that the user steers. It exists because generic resumes underperform against
specific postings and manual tailoring is slow and error-prone. Success is the
user downloading a tailored, professionally typeset PDF resume (and optional
cover letter) that they trust, having reviewed and approved the substantive
changes along the way.

The product is in **active development** heading toward a real launch with real
users; it is pre-launch. Marketing and landing surfaces must stay honest about
current capability and must not invent users, testimonials, metrics, or launch
claims.

## Positioning

The pipeline is a **step-by-step orchestrator (LangGraph) that pauses at named
human-in-the-loop gates**. Instead of a black-box "tailor my resume" button,
the flow advances through discrete, inspectable stages — compatibility score →
bullet selection → rewrite → CV assembly → cover letter — and stops at each gate
so the user can **approve, edit with feedback, or reject** before the next stage
runs. Each gate ships a typed `narration` and `preview` of exactly what is
being proposed. A neighboring "AI resume builder" cannot truthfully claim this
staged, reviewable, feedback-incorporating transformation; most competitors
return a finished document in one shot.

## Operating Context

- The user works in a browser session: choose a flow, upload a resume PDF,
  optionally add a job description (URL or pasted text), start analysis, then
  work through a chat-style interface where structured gate panels appear above
  a free-text input.
- Free-text chat during a pending gate is folded by the backend into that
  gate's "edit" feedback when the gate allows edits.
- Voice input is available in the chat: recordings are transcribed and
  auto-translated to English before insertion.
- Output PDFs are rendered from LaTeX templates via Tectonic (or `pdflatex`),
  so typographic quality of the generated documents is a product concern.
- Runs as a Next.js frontend proxying `/api/*` to a FastAPI backend
  (`localhost:8000`); backend calls OpenAI (GPT-4 class) and can be configured
  for Anthropic, Gemini, or Groq.

## Capabilities and Constraints

**Flows** (the `flow` concept in the codebase):

- `job_tailoring` — CV + job description in; produces a tailored CV and an
  optional cover letter. Fully implemented. Gate sequence includes: present
  compatibility score, approve bullet selection, approve rewrite, choose cover
  letter language, provide cover letter recipient, approve cover letter.
- `cv_review` — CV only; a structured quality review across header, education,
  experience, leadership, skills/projects, and an overall assessment, each with
  its own gate. No job posting is referenced in this flow.
- `discovery` — career discovery. Reserved in the orchestrator as a stub and
  surfaced in the UI as a disabled "Coming soon" option. Not yet available;
  future work must not present it as functional.

**Domain vocabulary** (keep consistent in UI and copy):

- "flow", "gate" (kinds: approval / choice / input), "step", "narration",
  "preview".
- Gate actions: approve, edit (with feedback), reject, choose.
- "CV" and "resume" are used interchangeably.
- German cover letters use "Anschreiben", "Betreff", "Grußformel".

**Constraints:**

- Resume upload: PDF only, max 10MB.
- Cover letters generate in English or German (user-selected per generation).
- CV extraction may return clarification questions that must be answered before
  the profile is complete.
- Generation is slow relative to web norms (CV generation ~20–40s); the UI must
  treat multi-second waits as a normal state, not an error.
- Long-running backend calls: the Next proxy timeout is set to 5 minutes.

**Explicitly undecided:**

- Authentication, accounts, and persistence of past sessions/documents are not
  established. Sessions are currently created per chat with no login.
- Whether `discovery` ships, and in what form.
- Pricing, licensing, and hosting/deployment model.

## Brand Commitments

- Name: **JobWriterAI** (one word; also exposed as `NEXT_PUBLIC_APP_NAME`).
  Note: `frontend/src/app/layout.tsx` metadata currently says "Resume AI Agent"
  — that is drift against this name, not a second brand.
- Source repository: `github.com/Valedicode/jobwriterai`.
- No logo, wordmark, tagline, color, type, or voice guideline has been made
  binding. These are open for visual work to establish.

## Evidence on Hand

- Working product: both `job_tailoring` and `cv_review` flows are implemented
  end to end with backend tests (`backend/tests/`).
- Real generated artifacts: LaTeX-rendered CV and cover letter PDFs
  (`backend/app/templates/latex/`, smoke scripts in `backend/scripts/`).
- Architecture documentation: `SYSTEM_OVERVIEW.md`, `README.md`,
  `frontend/docs/`.
- **No** customers, testimonials, case studies, press, usage metrics, or
  benchmark results exist. Future marketing/landing work must not fabricate any
  of these.

## Product Principles

1. **Show the work before committing it.** Every substantive transformation is
   previewed at a gate the user can approve, edit, or reject. The UI's job is to
   make what changed, and why, legible at a glance.
2. **Waiting is a first-class state.** Multi-second and multi-minute operations
   are expected. Progress, staged feedback, and "you can leave this open"
   reassurance matter more than raw speed.
3. **Two distinct jobs, one system.** "Review my CV" and "Tailor to a job" are
   different intents with different flows; never collapse them into a generic
   funnel or imply a job posting is required.
4. **Honest about maturity.** Pre-launch with no traction — surfaces speak to
   what the product does now, not to adoption it doesn't have. Unbuilt flows
   are marked unbuilt.
5. **The output is a document.** Typographic and structural quality of the
   generated PDF is part of the product, not an afterthought of the backend.

## Accessibility & Inclusion

Target **WCAG 2.1 AA**. Particular attention for this product: the chat + gate
interface must be fully keyboard operable, gate previews and diffs must not rely
on color alone to distinguish original from rewritten text, and long-running
operations must be announced to assistive tech rather than shown only as
spinners.
