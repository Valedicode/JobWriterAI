## JobWriterAI

JobWriterAI is a multi-agent system that takes your generic PDF resume and a specific Job Description URL, then uses a supervisor agent to coordinate specialized analysis agents and a writer agent: the analysis agents deeply understand your resume and the job description, and the writer agent uses that shared understanding to generate a refined, targeted resume and a matching cover letter.

---

## Concept

You provide:

- **Generic resume** as a PDF file
- **Job Description** as a URL (or pasted text, depending on the interface)

JobWriterAI's core agents:

- **Supervisor Agent**:
  - Orchestrates the full workflow
  - Delegates analysis tasks and decides when to (re)generate drafts
  - Ensures consistency between resume and cover letter
- **Resume Analysis Agent**:
  - Parses your existing resume from PDF
  - Extracts your experience, skills, and achievements
- **Job Description Analysis Agent**:
  - Scrapes and parses the job description
  - Identifies required skills, responsibilities, and keywords
- **Writer Agent**:
  - Combines the analyses into a **gap analysis** between JD requirements and your current resume
  - Iteratively **rewrites and optimizes** bullets, summary, and skills
  - **Pauses for your review and feedback** (human-in-the-loop)
  - Incorporates your feedback into the next iteration
  - Outputs a **clean, professional, targeted PDF resume**
  - Generates a **matching, targeted cover letter** aligned with the same requirements and style

---

## Example Workflow

1. Upload your **base resume PDF**.
2. Provide a **Job Description URL** (or paste the job description text).
3. The **Supervisor Agent** coordinates:
   - Scrapes/parses the JD  
   - Parses your resume  
   - Runs gap analysis and, via the Writer Agent, generates a first tailored draft  
4. Review a **diff view** of proposed changes:
   - Original vs rewritten bullets and sections  
   - Notes on which JD requirements are being satisfied  
5. Give feedback like:
   - “This looks too formal, make it punchier.”
6. The Writer Agent regenerates the draft with your style preferences.
7. When satisfied, click **Approve** to finalize your targeted resume.
8. Optionally, trigger the **Cover Letter Agent** to create a tailored cover letter using the approved resume and JD.
9. Download your **targeted PDF resume** (and cover letter, if generated).


## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.12 (recommended) or 3.11+ (managed by conda)
- [Tectonic](https://tectonic-typesetting.github.io/) for LaTeX-to-PDF export (recommended), or a TeX distribution with `pdflatex` (`LATEX_COMPILER=pdflatex`)

### Backend setup

1. **Clone the repo**

```bash
git clone https://github.com/Valedicode/jobwriterai.git
cd jobwriterai
```

2. **Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate jobwriterai
```

3. **Install backend Python dependencies**

```bash
pip install -r backend/requirements.txt
```

4. **Configure backend environment variables**

Create a `.env` file inside the `backend/` directory (next to `requirements.txt`).  
Use `backend/env.example` as a reference; at minimum you should set:

```bash
# backend/.env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here  (voice transcription, or LLM fallback)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GEMINI_API_KEY=your_gemini_api_key_here
# GROQ_API_KEY=your_groq_api_key_here
```

5. **Session storage (optional, recommended)**

Without `DATABASE_URL`, tailoring sessions live in memory and are lost whenever the backend restarts (including every `--reload`). To keep them, point the backend at a Postgres database; the checkpoint tables are created on startup:

```bash
# backend/.env
DATABASE_URL=postgresql://user:password@localhost:5432/jobwriterai
# SESSION_TTL_HOURS=24   # sessions and their CV data are deleted after this much inactivity
```

### Frontend setup

1. **Install frontend dependencies**

```bash
cd frontend
pnpm install
```

2. **Configure frontend environment variables**

Create a `.env.local` file in `frontend/` (based on `frontend/env.example` if present).

### Running the app

**Quick Start:**

**Terminal 1 - Start Backend:**

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Wait for: `Application startup complete.`

**Terminal 2 - Start Frontend:**

```bash
cd frontend
pnpm dev
# or: npm run dev
```

Wait for: `Ready on http://localhost:3000`

Then open `http://localhost:3000` in your browser.

---

