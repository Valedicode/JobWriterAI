"""
LaTeX rendering and PDF compilation for CV and cover letter export.

CV layout follows a Harvard-style two-column template driven by JSON Resume
taxonomy (see ``app.services.resume_adapter``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.services.resume_adapter import build_harvard_sections

TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "latex"

_GERMAN_MONTHS = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}


def latex_escape(text: str | None) -> str:
    """Escape user-provided text for safe inclusion in LaTeX."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "#": r"\#",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_url(text: str | None) -> str:
    """Escape a URL for use inside ``\\href{...}{...}``."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["latex_escape"] = latex_escape
    env.filters["latex_url"] = latex_url
    return env


# --------------------------------------------------------------------------- #
# CV page layout
#
# Page-break rules (the template enforces the first four, the PDF check the rest):
# - a heading never ends a page: it travels with its section's first entries;
# - an entry (title rows plus bullets) never splits across pages;
# - no section leaves a single entry on one side of a break: sections of up to
#   SMALL_SECTION_ENTRIES move as a whole, longer ones break only with at least
#   two entries on each side;
# - no stranded single lines (maximum widow / club penalties);
# - no short orphan section alone on the last page, and no nearly empty last
#   page: re-render compactly, then fold Languages into Skills, else flag it.
# --------------------------------------------------------------------------- #

CV_LAYOUTS: dict[str, dict[str, str]] = {
    # section_gap sits above a heading: inside the keep-together box the
    # center environment adds no space of its own, so the gap carries it.
    "normal": {"linespread": "0.96", "margin_x": "1.06cm", "margin_top": "0.8cm", "margin_bottom": "0.6cm",
               "section_gap": "10pt", "entry_gap": "4pt"},
    "compact": {"linespread": "0.92", "margin_x": "0.9cm", "margin_top": "0.6cm", "margin_bottom": "0.5cm",
                "section_gap": "6pt", "entry_gap": "3pt"},
}

SMALL_SECTION_ENTRIES = 3
SMALL_ITEMIZE_ENTRIES = 8
# A block taller than this many estimated lines may not fit a page; it is
# split into per-entry blocks instead (LaTeX can't break inside a block).
_MAX_BLOCK_LINES = 40
MIN_LAST_PAGE_FILL = 0.38
ORPHAN_SECTION_LINES = 4
# How many of the last sections the balancing pass tries to start on page 2.
BALANCE_CANDIDATES = 3

_STOPWORDS = {
    "english": {"the", "and", "with", "for", "of", "to", "in", "a", "on", "as", "by", "from"},
    "german": {"der", "die", "das", "und", "mit", "für", "von", "zu", "im", "ein", "eine", "bei", "auf", "als"},
}


def cv_content_language(cv_data: dict[str, Any]) -> str:
    """'german' when the CV's own sentences are mostly German, else 'english'.
    Section titles follow the content, so they never contradict it."""
    texts: list[str] = [str(cv_data.get("summary") or "")]
    for section, field in (("experience", "responsibilities"), ("projects", "outcomes"),
                           ("leadership_activities", "highlights")):
        for entry in cv_data.get(section) or []:
            if isinstance(entry, dict):
                texts.extend(str(t) for t in entry.get(field) or [])
    words = re.findall(r"[a-zäöüß]+", " ".join(texts).casefold())
    german = sum(w in _STOPWORDS["german"] for w in words)
    english = sum(w in _STOPWORDS["english"] for w in words)
    return "german" if german > english else "english"


def _estimated_lines(entry: dict[str, Any]) -> int:
    lines = len(entry.get("rows") or []) or 1
    for bullet in entry.get("bullets") or []:
        lines += 1 + len(bullet) // 110
    if entry.get("text"):
        lines += len(entry["text"]) // 110
    return lines


def _page_blocks(section: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a section into blocks LaTeX may only break between."""
    entries = section.get("entries") or []
    small = SMALL_ITEMIZE_ENTRIES if section.get("style") == "itemize" else SMALL_SECTION_ENTRIES
    total_lines = sum(_estimated_lines(e) for e in entries)
    if len(entries) <= small and total_lines <= _MAX_BLOCK_LINES:
        return [{"heading": True, "entries": entries}]
    if len(entries) <= 3:
        # Too tall to keep whole: heading with the first entry, then one block each.
        return [{"heading": True, "entries": entries[:1]}] + [{"heading": False, "entries": [e]} for e in entries[1:]]
    # At least two entries on each side of any break.
    middle = entries[2:-2]
    return (
        [{"heading": True, "entries": entries[:2]}]
        + [{"heading": False, "entries": [e]} for e in middle]
        + [{"heading": False, "entries": entries[-2:]}]
    )


def _block_anchor(section: dict[str, Any], block: dict[str, Any]) -> str:
    """The first text a block prints, used to find it in the rendered PDF."""
    if block["heading"]:
        return section["title"]
    entry = block["entries"][0] if block["entries"] else {}
    if entry.get("rows"):
        return entry["rows"][0].get("left", "")
    return entry.get("label") or entry.get("text", "")


def _cv_context(
    cv_data: dict[str, Any],
    section_order: list[str] | None,
    *,
    convention: str,
    merge_languages: bool,
    break_before_block: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Template context plus the anchor text of every block, in print order."""
    header, sections = build_harvard_sections(
        cv_data, section_order=section_order, language=cv_content_language(cv_data),
        convention=convention, merge_languages=merge_languages,
    )
    anchors: list[str] = []
    for section in sections:
        section["blocks"] = _page_blocks(section)
        for block in section["blocks"]:
            block["new_page"] = len(anchors) == break_before_block
            anchors.append(_block_anchor(section, block))
    return header, sections, anchors


def render_cv_tex(
    cv_data: dict[str, Any],
    section_order: list[str] | None = None,
    *,
    layout: str = "normal",
    convention: str = "international",
    merge_languages: bool = True,
    break_before_block: int | None = None,
) -> str:
    """Render a Harvard-style CV LaTeX document from structured CV JSON.
    ``break_before_block`` starts that block (in print order) on a new page."""
    header, sections, _ = _cv_context(
        cv_data, section_order, convention=convention,
        merge_languages=merge_languages, break_before_block=break_before_block,
    )
    env = _jinja_env()
    template = env.get_template("cv.tex.j2")
    return template.render(**header, sections=sections, layout=CV_LAYOUTS[layout])


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def _first_block_on_page_two(page_two_first_line: str, anchors: list[str]) -> int | None:
    """Index of the block that starts page 2, matched by its first text."""
    line = _normalized(page_two_first_line)
    if not line:
        return None
    for index, anchor in enumerate(anchors):
        key = _normalized(anchor)[:25]
        if key and line.startswith(key):
            return index
    return None


# Margins are at most ~1.4cm top + bottom; the usable height of a page is the
# page height minus this.
_MARGIN_ALLOWANCE_PT = 40.0
# Balancing must not leave page 1 half empty.
MIN_FIRST_PAGE_FILL = 0.5


def analyze_cv_pdf(pdf_path: Path) -> dict[str, Any]:
    """Pages, how full the first and last pages are (share of the usable page
    height), and how many text lines the last page holds."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        def lines(page: Any) -> list[tuple[float, float, str]]:
            found = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"])
                    if text.strip():
                        found.append((line["bbox"][1], line["bbox"][3], text))
            return sorted(found)

        first_lines, last_lines = lines(doc[0]), lines(doc[-1])
        second_lines = lines(doc[1]) if doc.page_count > 1 else []
        first = [(t, b) for t, b, _ in first_lines]
        last = [(t, b) for t, b, _ in last_lines]
        pages = doc.page_count
        usable = doc[0].rect.height - _MARGIN_ALLOWANCE_PT

    def fill(found: list[tuple[float, float]]) -> float:
        if not found:
            return 0.0
        return round(min((max(b for _, b in found) - min(t for t, _ in found)) / usable, 1.0), 3)

    return {"pages": pages, "first_page_fill": fill(first), "last_page_fill": fill(last),
            "last_page_lines": len({round(t) for t, _ in last}),
            "page_two_first_line": second_lines[0][2] if second_lines else ""}


def layout_problems(analysis: dict[str, Any]) -> list[str]:
    """What breaks the page rules, in the user's terms; empty when it's fine."""
    pages = analysis["pages"]
    if pages <= 1:
        return []
    problems = []
    if analysis["last_page_lines"] <= ORPHAN_SECTION_LINES:
        problems.append(f"the last page holds only {analysis['last_page_lines']} line(s)")
    elif analysis["last_page_fill"] < MIN_LAST_PAGE_FILL:
        problems.append(f"the last page is only {analysis['last_page_fill']:.0%} full")
    if analysis.get("first_page_fill", 1.0) < MIN_FIRST_PAGE_FILL:
        problems.append(f"the first page is only {analysis['first_page_fill']:.0%} full")
    if pages > 2:
        problems.append(f"the CV runs to {pages} pages")
    return problems


def render_cv_pdf(
    cv_data: dict[str, Any],
    output_dir: Path,
    job_name: str,
    *,
    section_order: list[str] | None = None,
    convention: str = "international",
) -> tuple[Path, dict[str, Any]]:
    """
    Compile the CV, check the page rules on the PDF, and re-render until they
    hold: first compact spacing, then whole blocks moved to page 2. Returns the
    PDF and a report ({pages, layout, page_break_before_block, problems});
    problems left after the last attempt are reported, not hidden.
    """
    results: list[tuple[tuple[str, bool, int | None], dict[str, Any], str]] = []

    def attempt(layout: str, merge: bool, break_before_block: int | None) -> bool:
        """Render and measure one variant; True when it keeps every rule."""
        tex = render_cv_tex(cv_data, section_order, layout=layout, convention=convention,
                            merge_languages=merge, break_before_block=break_before_block)
        name = f"{job_name}__try{len(results)}"
        analysis = analyze_cv_pdf(compile_tex_to_pdf(tex, output_dir, name))
        results.append(((layout, merge, break_before_block), analysis, name))
        return not layout_problems(analysis)

    # Spoken languages are a line in Skills (when there is a Skills section),
    # so a short Languages section can't be stranded on the last page.
    # 1. As designed; 2. compact, to fit on one page.
    done = attempt("normal", True, None) or attempt("compact", True, None)
    # 4. Balance two pages: move whole blocks (a section, or entries of a long
    # one) from the end of page 1 to page 2 by breaking one to three blocks
    # earlier than LaTeX did.
    if not done and results[0][1]["pages"] == 2:
        _, _, anchors = _cv_context(cv_data, section_order, convention=convention,
                                    merge_languages=True, break_before_block=None)
        first_on_two = _first_block_on_page_two(results[0][1]["page_two_first_line"], anchors)
        if first_on_two:
            for back in range(1, BALANCE_CANDIDATES + 1):
                if first_on_two - back <= 0 or attempt("normal", True, first_on_two - back):
                    break

    # The first attempt that keeps every rule, else the least bad one.
    def badness(result: tuple[Any, dict[str, Any], str]) -> tuple[int, int, float]:
        _, analysis, _ = result
        return len(layout_problems(analysis)), analysis["pages"], -analysis["last_page_fill"]

    best = next((r for r in results if not layout_problems(r[1])), None) or min(results, key=badness)
    (layout, merge, break_before), analysis, name = best
    analysis = {k: v for k, v in analysis.items() if k != "page_two_first_line"}
    final_pdf = output_dir / f"{job_name}.pdf"
    (output_dir / f"{name}.pdf").replace(final_pdf)
    for _, _, other in results:
        for suffix in (".pdf", ".tex", ".aux", ".log", ".out"):
            leftover = output_dir / f"{other}{suffix}"
            if leftover.exists() and leftover != final_pdf:
                leftover.unlink()
    return final_pdf, {
        **analysis,
        "layout": layout,
        "merged_languages": merge,
        "page_break_before_block": break_before,
        "attempts": len(results),
        "problems": layout_problems(analysis),
    }


def _format_date(language: str) -> str:
    now = datetime.now()
    if language.lower() == "german":
        return f"{now.day}. {_GERMAN_MONTHS[now.month]} {now.year}"
    return now.strftime("%B %d, %Y")


def _short_link_label(url: str, fallback: str) -> str:
    """Human-readable link text for the header column."""
    url = url.strip()
    if not url:
        return fallback
    for prefix in ("https://", "http://", "www."):
        if url.lower().startswith(prefix):
            url = url[len(prefix) :]
    return url.rstrip("/") or fallback


def _cover_letter_header_links(cv_data: dict[str, Any]) -> list[tuple[str, str]]:
    """Up to two (url, label) pairs for the header right column."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(url: str | None, label: str) -> None:
        if not isinstance(url, str) or not url.strip():
            return
        clean = url.strip()
        if clean in seen or len(links) >= 2:
            return
        seen.add(clean)
        links.append((clean, _short_link_label(clean, label)))

    _add(cv_data.get("linkedin_url"), "LinkedIn")
    _add(cv_data.get("github_url"), "GitHub")
    _add(cv_data.get("portfolio_url"), "Portfolio")
    return links


def extract_recipient_name(job_data: dict[str, Any] | None) -> str:
    """Return a named recipient from job posting data, or empty string."""
    if not isinstance(job_data, dict):
        return ""
    for key in ("recipient_name", "contact_name", "hiring_contact", "contact_person"):
        val = job_data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


_GENERIC_RECIPIENT_RE = re.compile(
    r"\b(team|hiring|recruit\w*|hr\b|human resources|personalabteilung|sir or madam|damen und herren)",
    re.IGNORECASE,
)
_FEMALE_TITLE_RE = re.compile(r"^(frau|ms\.?|mrs\.?|miss)\s+", re.IGNORECASE)
_MALE_TITLE_RE = re.compile(r"^(herr|mr\.?)\s+", re.IGNORECASE)


def is_generic_recipient(recipient: str) -> bool:
    """'Hiring Team', 'HR', 'Sir or Madam': no named person."""
    return bool(_GENERIC_RECIPIENT_RE.search(recipient or ""))


def _cover_letter_salutation(recipient_info: str, *, is_german: bool) -> str:
    """
    German: 'Sehr geehrte Frau Weber,' / 'Sehr geehrter Herr Weber,' when the
    title is given, else the neutral 'Guten Tag Alex Weber,' (no guessing a
    gender from a name); 'Sehr geehrte Damen und Herren,' for a team.
    English: 'Dear Ms Weber,' / 'Dear Alex Weber,' / 'Dear Hiring Team,'.
    """
    recipient = (recipient_info or "").strip()
    if not recipient:
        raise ValueError("recipient_info is required for the cover letter salutation")
    if is_german:
        if is_generic_recipient(recipient):
            return "Sehr geehrte Damen und Herren,"
        female, male = _FEMALE_TITLE_RE.match(recipient), _MALE_TITLE_RE.match(recipient)
        if female:
            return f"Sehr geehrte Frau {recipient[female.end():].strip()},"
        if male:
            return f"Sehr geehrter Herr {recipient[male.end():].strip()},"
        return f"Guten Tag {recipient},"
    return f"Dear {recipient},"


def _english_sign_off(recipient_info: str) -> str:
    """'Yours sincerely' to a named person, 'Yours faithfully' to a team."""
    return "Yours faithfully" if is_generic_recipient(recipient_info) else "Yours sincerely"


def _cover_letter_context(
    cv_data: dict[str, Any] | None,
    job_data: dict[str, Any] | None,
    applicant_name: str,
    applicant_contact: str = "",
) -> dict[str, Any]:
    """Extract header fields for the entry-level cover letter template."""
    cv = cv_data or {}
    job = job_data or {}

    phone = str(cv.get("phone") or "").strip()
    email = str(cv.get("email") or "").strip()
    if not email and applicant_contact and "@" in applicant_contact:
        # Legacy callers pass "email | phone" in applicant_contact.
        email = applicant_contact.split("|")[0].strip()
    if not phone and "|" in applicant_contact:
        parts = [p.strip() for p in applicant_contact.split("|")]
        if len(parts) > 1:
            phone = parts[1]

    target_job = ""
    for key in ("job_title", "title", "position"):
        val = job.get(key)
        if isinstance(val, str) and val.strip():
            target_job = val.strip()
            break

    header_links = _cover_letter_header_links(cv)
    link_1 = header_links[0] if len(header_links) > 0 else None
    link_2 = header_links[1] if len(header_links) > 1 else None

    return {
        "name": str(cv.get("name") or applicant_name or "Applicant").strip(),
        "phone": phone,
        "email": email,
        "target_job": target_job,
        "link_1_url": link_1[0] if link_1 else "",
        "link_1_label": link_1[1] if link_1 else "",
        "link_2_url": link_2[0] if link_2 else "",
        "link_2_label": link_2[1] if link_2 else "",
    }


def render_cover_letter_tex(
    content: dict[str, Any],
    applicant_name: str,
    applicant_contact: str = "",
    recipient_info: str = "",
    cv_data: dict[str, Any] | None = None,
    job_data: dict[str, Any] | None = None,
) -> str:
    """Render a cover letter LaTeX document from CoverLetterContent JSON."""
    language = (content.get("language") or "english").lower()
    is_german = language == "german"

    paragraphs = [
        content.get("opening_paragraph", ""),
        content.get("body_paragraph_1", ""),
        content.get("body_paragraph_2", ""),
    ]
    body_three = content.get("body_paragraph_3", "")
    if isinstance(body_three, str) and body_three.strip():
        paragraphs.append(body_three)
    paragraphs.append(content.get("closing_paragraph", ""))
    paragraphs = [p for p in paragraphs if isinstance(p, str) and p.strip()]

    header = _cover_letter_context(cv_data, job_data, applicant_name, applicant_contact)

    if is_german:
        salutation = _cover_letter_salutation(recipient_info, is_german=True)
        template_name = "cover_letter_de.tex.j2"
        extra = {
            "betreff": content.get("betreff", ""),
            "sign_off": content.get("grussformel", "Mit freundlichen Grüßen"),
            "salutation": salutation,
            "document_title": "ANSCHREIBEN",
            "date_label": "Datum:",
        }
    else:
        template_name = "cover_letter.tex.j2"
        extra = {
            "salutation": _cover_letter_salutation(recipient_info, is_german=False),
            "sign_off": _english_sign_off(recipient_info),
            "document_title": "COVER LETTER",
            "date_label": "Date:",
        }

    env = _jinja_env()
    template = env.get_template(template_name)
    return template.render(
        current_date=_format_date(language),
        paragraphs=paragraphs,
        **header,
        **extra,
    )


def _compiler_binary() -> str:
    compiler = os.getenv("LATEX_COMPILER", "tectonic").strip().lower()
    if compiler == "pdflatex":
        return shutil.which("pdflatex") or "pdflatex"
    custom = os.getenv("TECTONIC_PATH", "").strip()
    if custom:
        return custom
    return shutil.which("tectonic") or "tectonic"


def _run_tectonic(tex_path: Path, output_dir: Path) -> None:
    binary = _compiler_binary()
    result = subprocess.run(
        [binary, "-X", "compile", "--outdir", str(output_dir), str(tex_path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Tectonic compilation failed (exit {result.returncode}).\n{detail}"
        )


def _run_pdflatex(tex_path: Path, output_dir: Path) -> None:
    binary = _compiler_binary()
    command = [
        binary,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]
    last_error = ""
    for _ in range(2):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout or "").strip()
    log_path = output_dir / f"{tex_path.stem}.log"
    if log_path.exists():
        last_error = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise RuntimeError(f"pdflatex compilation failed.\n{last_error}")


def compile_tex_to_pdf(tex_content: str, output_dir: Path, job_name: str) -> Path:
    """
    Write ``tex_content`` to ``output_dir/{job_name}.tex`` and compile to PDF.

    Returns the path to the generated PDF file.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", job_name).strip("_") or "document"
    output_dir.mkdir(parents=True, exist_ok=True)

    tex_path = output_dir / f"{safe_name}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    compiler = os.getenv("LATEX_COMPILER", "tectonic").strip().lower()
    try:
        if compiler == "pdflatex":
            _run_pdflatex(tex_path, output_dir)
        else:
            _run_tectonic(tex_path, output_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "LaTeX compiler not found. Install Tectonic "
            "(https://tectonic-typesetting.github.io/) or set LATEX_COMPILER=pdflatex "
            "with a TeX distribution installed."
        ) from exc

    pdf_path = output_dir / f"{safe_name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"Compilation finished but PDF was not produced: {pdf_path.name}"
        )
    return pdf_path
