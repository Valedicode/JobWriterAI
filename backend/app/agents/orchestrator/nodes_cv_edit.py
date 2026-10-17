"""
Generating a finished document's PDF again after it was changed.

Once a PDF exists, its document (final_documents) can change two ways: the
chat proposes a change to one line (``chat_actions.propose_cv_edit``), which
waits at the confirm_change gate like every chat proposal, or the user saves
their own edits in the editor (app.api.documents). Either way the new document
is written to the state and the session routes here: the PDF is generated
again and the session goes back to where it was, a pending gate or the end.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from app.agents.orchestrator.final_documents import (
    COVER_LETTER_DOCUMENT,
    CV_DOCUMENT,
    final_cover_letter,
    final_cv_document,
    letter_applicant,
    rendered,
)
from app.agents.orchestrator.nodes_cv_review import render_reviewed_cv
from app.agents.orchestrator.nodes_job_tailoring import _narrate, render_cover_letter_document, render_tailored_cv
from app.agents.orchestrator.state import OrchestratorState

CV_RERENDER = "cv_rerender"
COVER_LETTER_RERENDER = "cover_letter_rerender"


def _back(state: OrchestratorState, update: dict[str, Any], narration: str) -> Command:
    return Command(
        update=_narrate({**update, "after_edit_goto": None}, narration),
        goto=state.get("after_edit_goto") or END,
    )


def cv_rerender(state: OrchestratorState, config: RunnableConfig) -> Command:
    document = final_cv_document(state)
    if document is None:
        return _back(state, {}, "There's no finished CV to generate again.")
    render = render_reviewed_cv if state.get("flow") == "cv_review" else render_tailored_cv
    pdf = render(document, state.get("generated_files") or [], config)
    if not pdf.filename:
        return _back(state, {}, f"Your CV is saved, but generating the PDF failed: {pdf.error}")

    document = rendered(document)
    update: dict[str, Any] = {CV_DOCUMENT: document, "generated_files": pdf.files}
    lines = [f"Updated your CV: {pdf.filename}.", *pdf.notes]
    # The cover letter prints the CV's name and contact details: when they
    # changed, the letter is generated again so the two still match.
    letter = final_cover_letter(state)
    if letter is not None and letter.get("applicant") != letter_applicant(document["cv"]):
        letter_pdf, letter = render_cover_letter_document(
            letter, {**state, CV_DOCUMENT: document, "generated_files": pdf.files}, config
        )
        if letter_pdf.filename:
            update.update({COVER_LETTER_DOCUMENT: letter, "generated_files": letter_pdf.files})
            lines.append(f"Updated your cover letter with the new contact details: {letter_pdf.filename}.")
        else:
            lines.append(f"Updating your cover letter with the new contact details failed: {letter_pdf.error}")
    return _back(state, update, "\n".join(lines))


def cover_letter_rerender(state: OrchestratorState, config: RunnableConfig) -> Command:
    document = final_cover_letter(state)
    if document is None:
        return _back(state, {}, "There's no finished cover letter to generate again.")
    pdf, document = render_cover_letter_document(document, state, config)
    if not pdf.filename:
        return _back(state, {}, f"Your cover letter is saved, but generating the PDF failed: {pdf.error}")
    return _back(
        state, {COVER_LETTER_DOCUMENT: document, "generated_files": pdf.files},
        f"Updated your cover letter: {pdf.filename}.",
    )
