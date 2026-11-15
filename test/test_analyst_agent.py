"""
Tests for analyst_agent.py (RAG-7)

Covers:
  - Returns analysis + need_more_evidence + followup_query
  - Empty evidence is handled gracefully
"""

import pytest
from unittest.mock import MagicMock, patch

from analyst_agent import analyst_node


class TestAnalystNode:

    def test_returns_all_required_fields(self):
        mock_llm = MagicMock()
        # data_analysis returns a string
        mock_llm.invoke.side_effect = [
            MagicMock(content="Model B is the best performer."),
            MagicMock(content='{"need_more_evidence": false, "query": ""}'),
        ]

        state = {
            "question": "Which model is best?",
            "evidence": [
                {"text": "Model B 95%", "doc_name": "p.pdf", "page": 1,
                 "chunk_id": "x", "score": None},
            ],
        }

        result = analyst_node(state, mock_llm)

        assert "analysis" in result
        assert "need_more_evidence" in result
        assert "followup_query" in result
        assert isinstance(result["analysis"], str)
        assert len(result["analysis"]) > 0

    def test_need_more_evidence_true(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="Partial analysis."),
            MagicMock(content='{"need_more_evidence": true, "query": "memory usage stats"}'),
        ]

        state = {
            "question": "Compare memory usage",
            "evidence": [
                {"text": "Model A is fast", "doc_name": "p.pdf", "page": 2,
                 "chunk_id": "y", "score": None},
            ],
        }

        result = analyst_node(state, mock_llm)
        assert result["need_more_evidence"] is True
        assert result["followup_query"] == "memory usage stats"

    def test_empty_evidence(self):
        mock_llm = MagicMock()

        state = {
            "question": "What is X?",
            "evidence": [],
        }

        result = analyst_node(state, mock_llm)
        assert result["need_more_evidence"] is True
        assert "No evidence" in result["analysis"]
        # LLM should not have been called
        mock_llm.invoke.assert_not_called()

    def test_missing_evidence_key(self):
        mock_llm = MagicMock()

        state = {"question": "What is X?"}

        result = analyst_node(state, mock_llm)
        assert result["need_more_evidence"] is True
