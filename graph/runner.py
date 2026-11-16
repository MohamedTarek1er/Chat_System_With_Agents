"""
graph/runner.py – Pipeline execution entry point.
"""

from graph.workflow import build_graph

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
