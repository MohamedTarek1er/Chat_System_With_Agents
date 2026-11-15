"""
orchestrator.py – LangGraph pipeline (RAG-5)

Graph layout:
    START ──► retriever ──► analyst ──► (need more?) ──► answer ──► END
                 ▲                          │
                 └──────── (loop) ──────────┘

Loop cap (MAX_LOOPS, default 3) prevents infinite cycling.  When the cap
is hit while the analyst still wants more evidence, evidence_complete is
set to False so the answer agent can qualify its response.

Entry point:
    run_pipeline(question, chat_history=None) -> dict
        {"answer": str, "sources": list[EvidenceItem], "evidence_complete": bool}
"""

from langgraph.graph import StateGraph, START, END

from llm_setup import llm
from state import State
from retriever_agent import retrieve
from analyst_agent import analyst_node
from answer_tools.answer_agent import answer_question

MAX_LOOPS = 3


# ── Nodes (flat functions) ────────────────────────────────────

def retriever_node(state: State) -> dict:
    """Retrieve evidence for the current query."""
    # On the first pass use the original question;
    # on subsequent loops use the analyst's followup_query.
    query = state.get("followup_query") or state["question"]
    chat_history = state.get("chat_history")

    new_evidence = retrieve(query, chat_history=chat_history, llm=llm)

    # Merge with any evidence already collected (dedup by chunk_id)
    existing = {e["chunk_id"]: e for e in state.get("evidence", [])}
    for item in new_evidence:
        existing.setdefault(item["chunk_id"], item)

    return {"evidence": list(existing.values())}


def analyst_node_wrapper(state: State) -> dict:
    """Run analyst tools on evidence."""
    return analyst_node(state, llm)


def answer_node(state: State) -> dict:
    """Produce the final answer."""
    evidence_complete = state.get("evidence_complete", True)
    result = answer_question(
        state["question"],
        state.get("analysis", ""),
        state.get("evidence", []),
        evidence_complete=evidence_complete,
    )
    return {"final_answer": result["answer"], "evidence_complete": result["evidence_complete"]}


# ── Routing ───────────────────────────────────────────────────

def after_analyst(state: State) -> str:
    """Decide whether to loop back for more evidence or answer.

    Called after bump_loop_count, so loop_count is already incremented.
    """
    loop_count = state.get("loop_count", 0)

    if state.get("need_more_evidence") and loop_count < MAX_LOOPS:
        return "retriever"

    return "answer"


def bump_loop_count(state: State) -> dict:
    """Increment the loop counter and set evidence_complete if capped."""
    loop_count = state.get("loop_count", 0) + 1
    updates = {"loop_count": loop_count}

    # If the analyst still wants more but we've hit the cap, flag it
    if state.get("need_more_evidence") and loop_count >= MAX_LOOPS:
        updates["evidence_complete"] = False

    return updates


# ── Graph assembly ────────────────────────────────────────────

def build_graph():
    graph = StateGraph(State)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node_wrapper)
    graph.add_node("bump_loop", bump_loop_count)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "bump_loop")
    graph.add_conditional_edges("bump_loop", after_analyst, {
        "retriever": "retriever",
        "answer": "answer",
    })
    graph.add_edge("answer", END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────

_compiled_graph = None


def run_pipeline(question, chat_history=None):
    """Run the full RAG pipeline.

    Args:
        question:     The user's question (str).
        chat_history: Optional list of {"role": ..., "content": ...} dicts.

    Returns:
        {"answer": str, "sources": list[EvidenceItem], "evidence_complete": bool}
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state = {
        "question": question,
        "chat_history": chat_history or [],
        "evidence": [],
        "loop_count": 0,
        "evidence_complete": True,
    }

    final_state = _compiled_graph.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", ""),
        "sources": final_state.get("evidence", []),
        "evidence_complete": final_state.get("evidence_complete", True),
    }
