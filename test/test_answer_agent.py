"""
Tests for answer_agent.py (RAG-9)

Covers:
  1. evidence_complete=False  -> answer admits limited evidence
  2. Empty analysis           -> safe fallback text used
  3. Inline [S1], [S2] markers map to the right source via format_citations
"""

import sys
import types
import pytest
from unittest.mock import patch, MagicMock

# Stub out llm_setup so response_formatter can import without langchain_openai
_llm_setup = types.ModuleType("llm_setup")
_llm_setup.llm = MagicMock()
sys.modules.setdefault("llm_setup", _llm_setup)

from answer_tools.answer_agent import answer_question


# ── Helpers ───────────────────────────────────────────────────

SAMPLE_EVIDENCE = [
    {"text": "Model A achieved 90% accuracy.", "doc_name": "research_1.pdf",
     "page": 3, "chunk_id": "a1", "score": None},
    {"text": "Model B achieved 95% accuracy.", "doc_name": "research_2.pdf",
     "page": 5, "chunk_id": "b1", "score": None},
]

SAMPLE_ANALYSIS = (
    "Model B achieved the highest accuracy at 95%."
)


# ── 1. evidence_complete flag ─────────────────────────────────

class TestEvidenceComplete:

    @patch("answer_tools.answer_agent.generate_response")
    def test_incomplete_evidence_noted_in_analysis(self, mock_gen):
        """When evidence_complete=False the answer dict says so."""
        mock_gen.return_value = "Model B is best [S1] [S2]."

        result = answer_question(
            "Which is best?",
            SAMPLE_ANALYSIS,
            SAMPLE_EVIDENCE,
            evidence_complete=False,
        )

        assert isinstance(result, dict)
        assert result["evidence_complete"] is False
        # The incomplete-evidence note should have been appended to analysis
        # before calling generate_response
        call_args = mock_gen.call_args
        analysis_sent = call_args[0][1]  # second positional arg
        assert "loop reached its cap" in analysis_sent.lower() or \
               "incomplete evidence" in analysis_sent.lower()

    @patch("answer_tools.answer_agent.generate_response")
    def test_complete_evidence(self, mock_gen):
        mock_gen.return_value = "Answer [S1]."
        result = answer_question(
            "q", SAMPLE_ANALYSIS, SAMPLE_EVIDENCE, evidence_complete=True
        )
        assert result["evidence_complete"] is True


# ── 2. Empty analysis guard ───────────────────────────────────

class TestEmptyAnalysis:

    @patch("answer_tools.answer_agent.generate_response")
    def test_none_analysis(self, mock_gen):
        mock_gen.return_value = "Some answer."
        result = answer_question("q", None, SAMPLE_EVIDENCE)
        assert isinstance(result, dict)
        assert "answer" in result
        # generate_response should have received a non-empty analysis string
        analysis_sent = mock_gen.call_args[0][1]
        assert len(analysis_sent.strip()) > 0

    @patch("answer_tools.answer_agent.generate_response")
    def test_whitespace_analysis(self, mock_gen):
        mock_gen.return_value = "Some answer."
        result = answer_question("q", "   \n  ", SAMPLE_EVIDENCE)
        analysis_sent = mock_gen.call_args[0][1]
        assert len(analysis_sent.strip()) > 0

    def test_no_evidence_returns_dict(self):
        result = answer_question("q", "analysis", [])
        assert isinstance(result, dict)
        assert "answer" in result
        assert "evidence" in result["answer"].lower() or \
               "couldn't find" in result["answer"].lower()


# ── 3. Both return paths are the same dict type ───────────────

class TestReturnType:

    @patch("answer_tools.answer_agent.generate_response")
    def test_normal_path_returns_dict(self, mock_gen):
        mock_gen.return_value = "Answer text."
        result = answer_question("q", SAMPLE_ANALYSIS, SAMPLE_EVIDENCE)
        assert isinstance(result, dict)
        assert "answer" in result
        assert "evidence_complete" in result

    def test_empty_evidence_path_returns_dict(self):
        result = answer_question("q", "analysis", [])
        assert isinstance(result, dict)
        assert "answer" in result
        assert "evidence_complete" in result


# ── 4. Citation mapping: [S1], [S2] → correct source ─────────

class TestCitationMapping:

    @patch("answer_tools.answer_agent.generate_response")
    def test_citations_map_to_correct_source(self, mock_gen):
        """[S1] should map to research_1.pdf p.3, [S2] to research_2.pdf p.5."""
        mock_gen.return_value = (
            "Model A scored 90% [S1]. Model B scored 95% [S2]."
        )

        result = answer_question("q", SAMPLE_ANALYSIS, SAMPLE_EVIDENCE)
        answer = result["answer"]

        # [S1] → research_1.pdf, p. 3
        assert "research_1.pdf" in answer
        assert "p. 3" in answer

        # [S2] → research_2.pdf, p. 5
        assert "research_2.pdf" in answer
        assert "p. 5" in answer

        # No unverified sources
        assert "[Unverified source]" not in answer

    @patch("answer_tools.answer_agent.generate_response")
    def test_unknown_citation_marked_unverified(self, mock_gen):
        """A citation [S99] that doesn't exist should be flagged."""
        mock_gen.return_value = "Some claim [S99]."
        result = answer_question("q", SAMPLE_ANALYSIS, SAMPLE_EVIDENCE)
        assert "[Unverified source]" in result["answer"]
