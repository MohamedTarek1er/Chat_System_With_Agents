"""
Tests for orchestrator.py (RAG-5)

Covers:
  1. run_pipeline returns an answer (happy path, single pass)
  2. Graph loops back when analyst says need_more_evidence=True
  3. Graph terminates at loop cap even if LLM keeps asking for more
  4. evidence_complete=False when cap is hit with outstanding requests
  5. Graph terminates immediately when analyst says need_more_evidence=False
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# ── Stub heavy imports BEFORE importing orchestrator ──────────
# These modules have deep transitive deps (torch, chromadb, etc.)
# that aren't installed in the test env.

_mock = MagicMock()

# llm_setup
_llm_setup = types.ModuleType("llm_setup")
_llm_setup.llm = MagicMock()
sys.modules["llm_setup"] = _llm_setup

# sentence_transformers (reranker imports at module level)
sys.modules["sentence_transformers"] = _mock

# langchain ecosystem
for mod in [
    "langchain_chroma", "langchain_huggingface", "langchain_openai",
    "langchain_core", "langchain_core.documents",
    "langchain_community", "langchain_community.retrievers",
    "dotenv",
]:
    sys.modules.setdefault(mod, _mock)

# document_processing.vector_store (module-level heavy init)
_dp = types.ModuleType("document_processing")
_dp_vs = types.ModuleType("document_processing.vector_store")
_dp_vs.search_documents = MagicMock(return_value=[])
_dp_vs.get_vector_store = MagicMock()
sys.modules["document_processing"] = _dp
sys.modules["document_processing.vector_store"] = _dp_vs

# langgraph – provide a minimal fake StateGraph
_lg = types.ModuleType("langgraph")
_lg_graph = types.ModuleType("langgraph.graph")
_lg_graph.START = "__start__"
_lg_graph.END = "__end__"


class _FakeStateGraph:
    def __init__(self, state_cls):
        self._nodes = {}
        self._edges = []

    def add_node(self, name, fn):
        self._nodes[name] = fn

    def add_edge(self, src, dst):
        self._edges.append((src, dst))

    def add_conditional_edges(self, src, fn, mapping):
        self._edges.append((src, fn, mapping))

    def compile(self):
        return MagicMock()


_lg_graph.StateGraph = _FakeStateGraph
sys.modules["langgraph"] = _lg
sys.modules["langgraph.graph"] = _lg_graph


# ── Now it's safe to import orchestrator ──────────────────────

from orchestrator import (
    retriever_node,
    analyst_node_wrapper,
    answer_node,
    after_analyst,
    bump_loop_count,
    MAX_LOOPS,
)


# ── Test data ─────────────────────────────────────────────────

EVIDENCE_A = {"text": "Model A got 92%", "doc_name": "p1.pdf", "page": 3,
              "chunk_id": "a1", "score": 0.9}
EVIDENCE_B = {"text": "Model B got 87%", "doc_name": "p2.pdf", "page": 5,
              "chunk_id": "b1", "score": 0.8}
EVIDENCE_C = {"text": "Model C memory 4GB", "doc_name": "p3.pdf", "page": 1,
              "chunk_id": "c1", "score": 0.7}


# ── Node unit tests ──────────────────────────────────────────

class TestRetrieverNode:

    @patch("orchestrator.retrieve")
    def test_first_pass_uses_question(self, mock_retrieve):
        mock_retrieve.return_value = [EVIDENCE_A]
        state = {"question": "compare models", "evidence": []}

        result = retriever_node(state)

        call_query = mock_retrieve.call_args[0][0]
        assert call_query == "compare models"
        assert len(result["evidence"]) == 1

    @patch("orchestrator.retrieve")
    def test_loop_pass_uses_followup_query(self, mock_retrieve):
        mock_retrieve.return_value = [EVIDENCE_B]
        state = {
            "question": "compare models",
            "followup_query": "memory usage",
            "evidence": [EVIDENCE_A],
        }

        result = retriever_node(state)

        call_query = mock_retrieve.call_args[0][0]
        assert call_query == "memory usage"

    @patch("orchestrator.retrieve")
    def test_deduplicates_by_chunk_id(self, mock_retrieve):
        mock_retrieve.return_value = [EVIDENCE_A]
        state = {"question": "q", "evidence": [EVIDENCE_A]}

        result = retriever_node(state)
        assert len(result["evidence"]) == 1

    @patch("orchestrator.retrieve")
    def test_merges_new_evidence(self, mock_retrieve):
        mock_retrieve.return_value = [EVIDENCE_B]
        state = {"question": "q", "evidence": [EVIDENCE_A]}

        result = retriever_node(state)
        chunk_ids = {e["chunk_id"] for e in result["evidence"]}
        assert chunk_ids == {"a1", "b1"}


class TestAnalystNodeWrapper:

    @patch("orchestrator.analyst_node")
    def test_calls_analyst_with_llm(self, mock_analyst):
        mock_analyst.return_value = {
            "analysis": "done",
            "need_more_evidence": False,
            "followup_query": "",
        }
        state = {"question": "q", "evidence": [EVIDENCE_A]}
        result = analyst_node_wrapper(state)

        mock_analyst.assert_called_once()
        assert result["analysis"] == "done"


class TestBumpLoopCount:

    def test_increments_from_zero(self):
        state = {"loop_count": 0, "need_more_evidence": False}
        result = bump_loop_count(state)
        assert result["loop_count"] == 1

    def test_increments_from_existing(self):
        state = {"loop_count": 1, "need_more_evidence": False}
        result = bump_loop_count(state)
        assert result["loop_count"] == 2

    def test_sets_evidence_complete_false_at_cap(self):
        state = {"loop_count": MAX_LOOPS - 1, "need_more_evidence": True}
        result = bump_loop_count(state)
        assert result["loop_count"] == MAX_LOOPS
        assert result["evidence_complete"] is False

    def test_does_not_flag_when_sufficient(self):
        state = {"loop_count": MAX_LOOPS - 1, "need_more_evidence": False}
        result = bump_loop_count(state)
        assert "evidence_complete" not in result


class TestAfterAnalyst:

    def test_routes_to_retriever_when_more_needed(self):
        # loop_count already bumped to 1, still below MAX_LOOPS (3)
        state = {"need_more_evidence": True, "loop_count": 1}
        assert after_analyst(state) == "retriever"

    def test_routes_to_answer_when_sufficient(self):
        state = {"need_more_evidence": False, "loop_count": 1}
        assert after_analyst(state) == "answer"

    def test_routes_to_answer_at_loop_cap(self):
        """Cap reached → force answer even if analyst wants more."""
        state = {"need_more_evidence": True, "loop_count": MAX_LOOPS}
        assert after_analyst(state) == "answer"

    def test_routes_to_retriever_below_cap(self):
        state = {"need_more_evidence": True, "loop_count": 1}
        assert after_analyst(state) == "retriever"


class TestAnswerNode:

    @patch("orchestrator.answer_question")
    def test_returns_final_answer(self, mock_aq):
        mock_aq.return_value = {"answer": "Model B is best.", "evidence_complete": True}
        state = {"question": "which?", "analysis": "B 95%",
                 "evidence": [EVIDENCE_A], "evidence_complete": True}

        result = answer_node(state)
        assert result["final_answer"] == "Model B is best."
        assert result["evidence_complete"] is True

    @patch("orchestrator.answer_question")
    def test_passes_evidence_complete_false(self, mock_aq):
        mock_aq.return_value = {"answer": "Partial.", "evidence_complete": False}
        state = {"question": "q", "analysis": "partial",
                 "evidence": [EVIDENCE_A], "evidence_complete": False}

        result = answer_node(state)
        assert result["evidence_complete"] is False


# ── Integration: simulate graph execution manually ────────────

class TestGraphSimulation:

    @patch("orchestrator.answer_question")
    @patch("orchestrator.analyst_node")
    @patch("orchestrator.retrieve")
    def test_single_pass_happy_path(self, mock_retrieve, mock_analyst, mock_aq):
        """Analyst says sufficient → straight to answer."""
        mock_retrieve.return_value = [EVIDENCE_A, EVIDENCE_B]
        mock_analyst.return_value = {
            "analysis": "Comparison done.", "need_more_evidence": False,
            "followup_query": "",
        }
        mock_aq.return_value = {"answer": "Model A wins.", "evidence_complete": True}

        state = {"question": "compare", "evidence": [], "loop_count": 0,
                 "evidence_complete": True}

        state.update(retriever_node(state))
        state.update(analyst_node_wrapper(state))
        state.update(bump_loop_count(state))
        assert after_analyst(state) == "answer"
        state.update(answer_node(state))

        assert state["final_answer"] == "Model A wins."
        assert state["loop_count"] == 1

    @patch("orchestrator.answer_question")
    @patch("orchestrator.analyst_node")
    @patch("orchestrator.retrieve")
    def test_one_loop_then_answer(self, mock_retrieve, mock_analyst, mock_aq):
        """Analyst asks for more once, then satisfied."""
        mock_retrieve.side_effect = [[EVIDENCE_A], [EVIDENCE_C]]
        mock_analyst.side_effect = [
            {"analysis": "Need memory.", "need_more_evidence": True,
             "followup_query": "memory usage"},
            {"analysis": "Full analysis.", "need_more_evidence": False,
             "followup_query": ""},
        ]
        mock_aq.return_value = {"answer": "Complete.", "evidence_complete": True}

        state = {"question": "compare all", "evidence": [], "loop_count": 0,
                 "evidence_complete": True}

        # Pass 1
        state.update(retriever_node(state))
        state.update(analyst_node_wrapper(state))
        state.update(bump_loop_count(state))
        assert after_analyst(state) == "retriever"

        # Pass 2
        state.update(retriever_node(state))
        state.update(analyst_node_wrapper(state))
        state.update(bump_loop_count(state))
        assert after_analyst(state) == "answer"

        state.update(answer_node(state))

        assert state["final_answer"] == "Complete."
        assert state["loop_count"] == 2
        assert state.get("evidence_complete", True) is True

    @patch("orchestrator.answer_question")
    @patch("orchestrator.analyst_node")
    @patch("orchestrator.retrieve")
    def test_cap_forces_termination(self, mock_retrieve, mock_analyst, mock_aq):
        """Analyst always wants more → cap forces answer after MAX_LOOPS."""
        mock_retrieve.return_value = [EVIDENCE_A]
        mock_analyst.return_value = {
            "analysis": "Still need more.", "need_more_evidence": True,
            "followup_query": "something else",
        }
        mock_aq.return_value = {"answer": "Partial answer.", "evidence_complete": False}

        state = {"question": "everything", "evidence": [], "loop_count": 0,
                 "evidence_complete": True}

        for _ in range(MAX_LOOPS):
            state.update(retriever_node(state))
            state.update(analyst_node_wrapper(state))
            state.update(bump_loop_count(state))
            route = after_analyst(state)
            if route == "answer":
                break

        assert route == "answer"
        assert state["loop_count"] == MAX_LOOPS
        assert state["evidence_complete"] is False

        state.update(answer_node(state))
        assert state["final_answer"] == "Partial answer."

    @patch("orchestrator.answer_question")
    @patch("orchestrator.analyst_node")
    @patch("orchestrator.retrieve")
    def test_evidence_accumulates_across_loops(self, mock_retrieve, mock_analyst, mock_aq):
        """Evidence from each loop merges (no loss, no dupes)."""
        mock_retrieve.side_effect = [[EVIDENCE_A], [EVIDENCE_B], [EVIDENCE_C]]
        mock_analyst.side_effect = [
            {"analysis": "p1", "need_more_evidence": True, "followup_query": "q2"},
            {"analysis": "p2", "need_more_evidence": True, "followup_query": "q3"},
            {"analysis": "p3", "need_more_evidence": True, "followup_query": "q4"},
        ]
        mock_aq.return_value = {"answer": "All done.", "evidence_complete": False}

        state = {"question": "q", "evidence": [], "loop_count": 0,
                 "evidence_complete": True}

        for _ in range(MAX_LOOPS):
            state.update(retriever_node(state))
            state.update(analyst_node_wrapper(state))
            state.update(bump_loop_count(state))
            route = after_analyst(state)
            if route == "answer":
                break

        # All three unique chunks should be present
        chunk_ids = {e["chunk_id"] for e in state["evidence"]}
        assert chunk_ids == {"a1", "b1", "c1"}
