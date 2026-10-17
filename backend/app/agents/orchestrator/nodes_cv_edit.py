"""
Editing the finished CV from the chat.

Once a CV PDF exists, the chat can change one of its lines
(``chat_actions.propose_cv_edit``). Like every chat proposal the change waits at
the confirm_change gate. Approving it writes the new line into the CV the PDF
was made from and routes here: the PDF is generated again and the session goes
back to where it was, a pending gate or the end.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from app.agents.orchestrator.nodes_cv_review import render_reviewed_cv
from app.agents.orchestrator.nodes_job_tailoring import _narrate, render_tailored_cv
from app.agents.orchestrator.state import OrchestratorState

CV_RERENDER = "cv_rerender"


def cv_rerender(state: OrchestratorState, config: RunnableConfig) -> Command:
    if state.get("flow") == "cv_review":
        rendered = render_reviewed_cv(state.get("cv_data") or {}, state, config)
    else:
        rendered = render_tailored_cv(state, config)
    narration = (
        "\n".join([f"Updated your CV: {rendered.filename}.", *rendered.notes]) if rendered.filename
        else f"I changed the line, but generating the PDF failed: {rendered.error}"
    )
    return Command(
        update=_narrate({"generated_files": rendered.files, "after_edit_goto": None}, narration),
        goto=state.get("after_edit_goto") or END,
    )
