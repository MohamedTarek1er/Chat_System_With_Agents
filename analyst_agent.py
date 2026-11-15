"""
analyst_agent.py – Analyst Agent (RAG-7)

Runs analyst tools on retrieved evidence and produces:
  - analysis:           str   (LLM-generated analysis of the evidence)
  - need_more_evidence: bool  (from evidence_request)
  - query:              str   (follow-up retrieval query, empty if sufficient)

Tool-selection strategy: FIXED ORDER
  1. data_analysis  – always runs first to produce the core analysis.
  2. evidence_request – always runs second to decide if more evidence is needed.

Rationale: every question needs both steps, and the order is deterministic.
An LLM-chosen approach would add latency and token cost for no gain here
because both tools must fire on every call.  If the team later wants the
LLM to pick from a larger toolset (e.g. calculator, table_extractor,
document_comparison), switch to `llm.bind_tools(tools)` with a ToolNode
loop — but that decision is still open.
"""

from analyst_tools.data_analysis import data_analysis
from analyst_tools.evidence_request import evidence_request


def analyst_node(state, llm):
    """LangGraph-compatible analyst node.

    Reads: state["question"], state["evidence"]
    Returns: dict with analysis, need_more_evidence, followup_query
    """
    question = state["question"]
    evidence = state.get("evidence", [])

    if not evidence:
        return {
            "analysis": "No evidence was provided for analysis.",
            "need_more_evidence": True,
            "followup_query": question,
        }

    # Step 1 – Analyze the evidence
    analysis = data_analysis(llm, question, evidence)

    # Step 2 – Decide whether more evidence is needed
    er = evidence_request(llm, question, evidence)

    return {
        "analysis": analysis,
        "need_more_evidence": er["need_more_evidence"],
        "followup_query": er["query"],
    }
