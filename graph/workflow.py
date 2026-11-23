"""
graph/workflow.py – StateGraph definition, nodes, and conditional routing.
"""

from langgraph.graph import StateGraph, START, END

from core.llm import llm
from core.config import MAX_LOOPS
from graph.state import State
from agents.retriever.retriever_agent import retrieve
from agents.analyst.analyst_agent import make_analyst_node
from agents.answer.answer_agent import answer_question


def retriever_node(state: State) -> dict:
    """Retrieve evidence for the current query."""
    query = state.get("followup_query") or state["question"]
    chat_history = state.get("chat_history")

    new_evidence = retrieve(query, chat_history=chat_history, llm=llm)

    # Merge with any evidence already collected (dedup by chunk_id)
    existing = {e["chunk_id"]: e for e in state.get("evidence", [])}
    for item in new_evidence:
        existing.setdefault(item["chunk_id"], item)

    return {"evidence": list(existing.values())}


def answer_node(state: State) -> dict:
    """Produce the final answer."""
    evidence_complete = state.get("evidence_complete", True)
    result = answer_question(
        state["question"],
        state.get("analysis", ""),
        state.get("evidence", []),
        evidence_complete=evidence_complete,
    )
    return {
        "final_answer": result["answer"],
        "evidence_complete": result["evidence_complete"],
    }


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

    if state.get("need_more_evidence") and loop_count >= MAX_LOOPS:
        updates["evidence_complete"] = False

    return updates


def build_graph():
    """Assemble and compile the LangGraph agent workflow."""
    graph = StateGraph(State)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", make_analyst_node(llm))
    graph.add_node("bump_loop", bump_loop_count)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "bump_loop")
    graph.add_conditional_edges(
        "bump_loop",
        after_analyst,
        {
            "retriever": "retriever",
            "answer": "answer",
        },
    )
    graph.add_edge("answer", END)

    return graph.compile()
