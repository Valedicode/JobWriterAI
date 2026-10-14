"""
Orchestrator graph builder.

Compiles a single StateGraph with three entry branches (job_tailoring,
cv_review, discovery). The job-tailoring branch is fully implemented; the
cv_review and discovery branches are single-node stubs that will be expanded
in later migration steps without changing the API surface.

``build_graph(checkpointer=None)`` returns a compiled graph bound to the
supplied checkpointer (or the global one from
``app.services.graph_checkpointer.get_checkpointer`` if omitted). Callers run
the graph as:

    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke({"flow": "job_tailoring", ...}, config=config)
    # later, to resume after a gate:
    result = graph.invoke(Command(resume=resolution_dict), config=config)
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.orchestrator.nodes_cv_review import (
    CR_ENTRY,
    CR_GATE_ASSESSMENT,
    CR_GATE_EDUCATION,
    CR_GATE_EXPERIENCE,
    CR_GATE_HEADER,
    CR_GATE_LEADERSHIP,
    CR_GATE_SKILLS_PROJECTS,
    CR_STEP1_HEADER,
    CR_STEP2_EDUCATION,
    CR_STEP3_EXPERIENCE,
    CR_STEP4_LEADERSHIP,
    CR_STEP5_SKILLS_PROJECTS,
    CR_STEP6_ASSESSMENT,
    CR_STEP7_FINALIZE,
    cr_entry,
    cr_gate_assessment,
    cr_gate_education,
    cr_gate_experience,
    cr_gate_header,
    cr_gate_leadership,
    cr_gate_skills_projects,
    cr_step1_header,
    cr_step2_education,
    cr_step3_experience,
    cr_step4_leadership,
    cr_step5_skills_projects,
    cr_step6_assessment,
    cr_step7_finalize,
)
from app.agents.orchestrator.nodes_discovery import DISC_STUB, discovery_stub
from app.agents.orchestrator.nodes_job_tailoring import (
    JT_ENTRY,
    JT_GATE_APPROVE_COVER_LETTER,
    JT_GATE_APPROVE_REWRITE,
    JT_GATE_APPROVE_SELECTION,
    JT_GATE_CONFIRM_CHANGE,
    JT_GATE_CONFIRM_ELIGIBILITY,
    JT_GATE_COVER_LETTER,
    JT_GATE_COVER_LETTER_RECIPIENT,
    JT_STEP_SET_COVER_LETTER_RECIPIENT,
    JT_GATE_PRESENT_SCORE,
    JT_GATE_STEP_FAILED,
    JT_GATE_VERIFIER_FAILED,
    JT_STEP1_SUMMARIZE,
    JT_STEP2_COMPATIBILITY,
    JT_STEP3_STRATEGY,
    JT_STEP4_SELECT,
    JT_STEP5_REWRITE,
    JT_STEP5B_VERIFY,
    JT_STEP6_ASSEMBLE,
    JT_STEP7_GENERATE_CV_FILES,
    JT_STEP8_COVER_LETTER_CONTENT,
    JT_STEP9_COVER_LETTER_FILES,
    jt_entry,
    jt_gate_approve_cover_letter,
    jt_gate_approve_rewrite,
    jt_gate_approve_selection,
    jt_gate_confirm_change,
    jt_gate_confirm_eligibility,
    jt_gate_cover_letter,
    jt_gate_cover_letter_recipient,
    jt_gate_present_score,
    jt_gate_step_failed,
    jt_gate_verifier_failed,
    jt_step1_summarize_job,
    jt_step2_compute_compatibility,
    jt_step3_decide_strategy,
    jt_step4_select_prioritize,
    jt_step5_rewrite_enhance,
    jt_step5b_verify,
    jt_step6_assemble_cv,
    jt_step7_generate_cv_files,
    jt_step8_cover_letter_content,
    jt_step9_cover_letter_files,
    jt_step_set_cover_letter_recipient,
    route_after_rewrite,
    route_after_score,
    route_after_select,
)
from app.agents.orchestrator.state import OrchestratorState
from app.services.graph_checkpointer import get_checkpointer


# --------------------------------------------------------------------------- #
# Flow router
# --------------------------------------------------------------------------- #

def _route_by_flow(state: OrchestratorState) -> Literal[
    "jt_entry", "cr_entry", "discovery_stub"
]:
    flow = state.get("flow")
    if flow == "job_tailoring":
        return JT_ENTRY
    if flow == "cv_review":
        return CR_ENTRY
    return DISC_STUB


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """
    Build and compile the orchestrator StateGraph.

    The compiled graph is safe to cache and reuse across requests; thread
    isolation is provided by the checkpointer keyed on ``thread_id``.
    """
    graph = StateGraph(OrchestratorState)

    # --- branch entry stubs --------------------------------------------------
    graph.add_node(JT_ENTRY, jt_entry)
    graph.add_node(CR_ENTRY, cr_entry)
    graph.add_node(DISC_STUB, discovery_stub)

    # --- cv-review step + gate nodes ----------------------------------------
    graph.add_node(CR_STEP1_HEADER, cr_step1_header)
    graph.add_node(CR_GATE_HEADER, cr_gate_header)
    graph.add_node(CR_STEP2_EDUCATION, cr_step2_education)
    graph.add_node(CR_GATE_EDUCATION, cr_gate_education)
    graph.add_node(CR_STEP3_EXPERIENCE, cr_step3_experience)
    graph.add_node(CR_GATE_EXPERIENCE, cr_gate_experience)
    graph.add_node(CR_STEP4_LEADERSHIP, cr_step4_leadership)
    graph.add_node(CR_GATE_LEADERSHIP, cr_gate_leadership)
    graph.add_node(CR_STEP5_SKILLS_PROJECTS, cr_step5_skills_projects)
    graph.add_node(CR_GATE_SKILLS_PROJECTS, cr_gate_skills_projects)
    graph.add_node(CR_STEP6_ASSESSMENT, cr_step6_assessment)
    graph.add_node(CR_GATE_ASSESSMENT, cr_gate_assessment)
    graph.add_node(CR_STEP7_FINALIZE, cr_step7_finalize)

    # --- job-tailoring step nodes --------------------------------------------
    graph.add_node(JT_STEP1_SUMMARIZE, jt_step1_summarize_job)
    graph.add_node(JT_STEP2_COMPATIBILITY, jt_step2_compute_compatibility)
    graph.add_node(JT_STEP3_STRATEGY, jt_step3_decide_strategy)
    graph.add_node(JT_STEP4_SELECT, jt_step4_select_prioritize)
    graph.add_node(JT_STEP5_REWRITE, jt_step5_rewrite_enhance)
    # Fact check: routes via Command to JT_GATE_APPROVE_REWRITE, back to
    # JT_STEP5_REWRITE (one retry), or to JT_GATE_VERIFIER_FAILED.
    graph.add_node(JT_STEP5B_VERIFY, jt_step5b_verify)
    graph.add_node(JT_STEP6_ASSEMBLE, jt_step6_assemble_cv)
    graph.add_node(JT_STEP7_GENERATE_CV_FILES, jt_step7_generate_cv_files)
    graph.add_node(JT_STEP_SET_COVER_LETTER_RECIPIENT, jt_step_set_cover_letter_recipient)
    graph.add_node(JT_GATE_COVER_LETTER_RECIPIENT, jt_gate_cover_letter_recipient)
    graph.add_node(JT_STEP8_COVER_LETTER_CONTENT, jt_step8_cover_letter_content)
    graph.add_node(JT_STEP9_COVER_LETTER_FILES, jt_step9_cover_letter_files)

    # --- job-tailoring gate nodes (return Command(goto=...)) ----------------
    graph.add_node(JT_GATE_PRESENT_SCORE, jt_gate_present_score)
    graph.add_node(JT_GATE_APPROVE_SELECTION, jt_gate_approve_selection)
    graph.add_node(JT_GATE_APPROVE_REWRITE, jt_gate_approve_rewrite)
    graph.add_node(JT_GATE_COVER_LETTER, jt_gate_cover_letter)
    graph.add_node(JT_GATE_APPROVE_COVER_LETTER, jt_gate_approve_cover_letter)
    graph.add_node(JT_GATE_STEP_FAILED, jt_gate_step_failed)
    graph.add_node(JT_GATE_VERIFIER_FAILED, jt_gate_verifier_failed)
    graph.add_node(JT_GATE_CONFIRM_ELIGIBILITY, jt_gate_confirm_eligibility)
    # Reached only through a chat proposal (app.agents.orchestrator.chat_actions),
    # which redirects the paused graph here; it routes on via Command.
    graph.add_node(JT_GATE_CONFIRM_CHANGE, jt_gate_confirm_change)

    # --- entry routing -------------------------------------------------------
    graph.add_conditional_edges(
        START,
        _route_by_flow,
        {
            JT_ENTRY: JT_ENTRY,
            CR_ENTRY: CR_ENTRY,
            DISC_STUB: DISC_STUB,
        },
    )

    # --- linear edges for non-routing step nodes ----------------------------
    graph.add_edge(JT_ENTRY, JT_STEP1_SUMMARIZE)
    graph.add_edge(JT_STEP1_SUMMARIZE, JT_STEP2_COMPATIBILITY)
    # Step 2 asks about eligibility the CV is silent on (once per run);
    # JT_GATE_CONFIRM_ELIGIBILITY then goes to JT_STEP3_STRATEGY via Command.
    graph.add_conditional_edges(
        JT_STEP2_COMPATIBILITY,
        route_after_score,
        {JT_GATE_CONFIRM_ELIGIBILITY: JT_GATE_CONFIRM_ELIGIBILITY, JT_STEP3_STRATEGY: JT_STEP3_STRATEGY},
    )
    # The strategist's plan is reviewed together with the score.
    graph.add_edge(JT_STEP3_STRATEGY, JT_GATE_PRESENT_SCORE)
    # JT_GATE_PRESENT_SCORE returns Command(goto=JT_STEP4_SELECT | JT_STEP3_STRATEGY (edit) | END)
    # Steps 4 and 5 go to their review gate, or to JT_GATE_STEP_FAILED when
    # their tool reported an error. That gate returns Command(goto=<failed
    # step> | END), so a retry passes through the review gate again.
    graph.add_conditional_edges(
        JT_STEP4_SELECT,
        route_after_select,
        {JT_GATE_APPROVE_SELECTION: JT_GATE_APPROVE_SELECTION, JT_GATE_STEP_FAILED: JT_GATE_STEP_FAILED},
    )
    # JT_GATE_APPROVE_SELECTION returns Command(goto=JT_STEP5_REWRITE | JT_STEP4_SELECT | END)
    graph.add_conditional_edges(
        JT_STEP5_REWRITE,
        route_after_rewrite,
        {JT_STEP5B_VERIFY: JT_STEP5B_VERIFY, JT_GATE_STEP_FAILED: JT_GATE_STEP_FAILED},
    )
    # JT_GATE_APPROVE_REWRITE returns Command(goto=JT_STEP6_ASSEMBLE | JT_STEP5_REWRITE | END)
    graph.add_edge(JT_STEP6_ASSEMBLE, JT_STEP7_GENERATE_CV_FILES)
    graph.add_edge(JT_STEP7_GENERATE_CV_FILES, JT_GATE_COVER_LETTER)
    # JT_GATE_COVER_LETTER returns Command(goto=JT_STEP_SET_COVER_LETTER_RECIPIENT | END)
    # JT_STEP_SET_COVER_LETTER_RECIPIENT / JT_GATE_COVER_LETTER_RECIPIENT route via Command
    graph.add_edge(JT_STEP8_COVER_LETTER_CONTENT, JT_GATE_APPROVE_COVER_LETTER)
    # JT_GATE_APPROVE_COVER_LETTER returns Command(goto=JT_STEP9_COVER_LETTER_FILES | JT_STEP8... | END)
    graph.add_edge(JT_STEP9_COVER_LETTER_FILES, END)

    # --- cv-review linear edges --------------------------------------------
    # Step nodes always flow into their gate. Gates return Command(goto=...)
    # routing to the next step, looping back on edit, or ending on reject.
    graph.add_edge(CR_ENTRY, CR_STEP1_HEADER)
    graph.add_edge(CR_STEP1_HEADER, CR_GATE_HEADER)
    graph.add_edge(CR_STEP2_EDUCATION, CR_GATE_EDUCATION)
    graph.add_edge(CR_STEP3_EXPERIENCE, CR_GATE_EXPERIENCE)
    graph.add_edge(CR_STEP4_LEADERSHIP, CR_GATE_LEADERSHIP)
    graph.add_edge(CR_STEP5_SKILLS_PROJECTS, CR_GATE_SKILLS_PROJECTS)
    graph.add_edge(CR_STEP6_ASSESSMENT, CR_GATE_ASSESSMENT)
    # CR_GATE_ASSESSMENT routes to CR_STEP7_FINALIZE on approve, END on reject.
    graph.add_edge(CR_STEP7_FINALIZE, END)

    # --- discovery stub ends immediately ------------------------------------
    graph.add_edge(DISC_STUB, END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())
