"""
Tests for evidence_request.py (RAG-8)

Covers:
  1. Valid JSON response
  2. JSON wrapped in ```json code fences
  3. Garbage / unparseable output  ->  safe fallback
  4. _format_evidence produces clean text, not str(dict)
"""

import pytest
from unittest.mock import MagicMock

from analyst_tools.evidence_request import (
    evidence_request,
    _parse_json_safe,
    _format_evidence,
)


# ── _parse_json_safe ──────────────────────────────────────────

class TestParseJsonSafe:

    def test_valid_json_need_more(self):
        raw = '{"need_more_evidence": true, "query": "memory usage"}'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": True, "query": "memory usage"}

    def test_valid_json_sufficient(self):
        raw = '{"need_more_evidence": false, "query": ""}'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": False, "query": ""}

    def test_code_fence_json(self):
        raw = '```json\n{"need_more_evidence": true, "query": "latency stats"}\n```'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": True, "query": "latency stats"}

    def test_code_fence_no_lang(self):
        raw = '```\n{"need_more_evidence": false, "query": ""}\n```'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": False, "query": ""}

    def test_garbage_returns_fallback(self):
        result = _parse_json_safe("Sure! Here is the analysis of your documents...")
        assert result == {"need_more_evidence": False, "query": ""}

    def test_empty_string_returns_fallback(self):
        result = _parse_json_safe("")
        assert result == {"need_more_evidence": False, "query": ""}

    def test_missing_key_returns_fallback(self):
        raw = '{"query": "something"}'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": False, "query": ""}

    def test_array_returns_fallback(self):
        raw = '[1, 2, 3]'
        result = _parse_json_safe(raw)
        assert result == {"need_more_evidence": False, "query": ""}


# ── _format_evidence ──────────────────────────────────────────

class TestFormatEvidence:

    def test_evidence_item_dict(self):
        items = [
            {"text": "Model A got 92%", "doc_name": "paper.pdf", "page": 3,
             "chunk_id": "abc", "score": None},
        ]
        out = _format_evidence(items)
        assert "paper.pdf" in out
        assert "p. 3" in out
        assert "Model A got 92%" in out
        # Must NOT contain raw dict repr
        assert "chunk_id" not in out
        assert "'text'" not in out

    def test_page_none_omitted(self):
        items = [
            {"text": "some text", "doc_name": "file.txt", "page": None,
             "chunk_id": "x", "score": None},
        ]
        out = _format_evidence(items)
        assert "p. " not in out
        assert "file.txt" in out


# ── evidence_request (integration-style with mock LLM) ───────

class TestEvidenceRequest:

    def test_returns_parsed_result(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"need_more_evidence": true, "query": "memory usage"}'
        )
        evidence = [
            {"text": "Model A 92%", "doc_name": "p.pdf", "page": 1,
             "chunk_id": "a", "score": None},
        ]
        result = evidence_request(mock_llm, "compare models", evidence)
        assert result["need_more_evidence"] is True
        assert result["query"] == "memory usage"

    def test_garbage_llm_output_falls_back(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I cannot determine this from the evidence."
        )
        evidence = [
            {"text": "data", "doc_name": "f.pdf", "page": 2,
             "chunk_id": "b", "score": None},
        ]
        result = evidence_request(mock_llm, "question", evidence)
        assert result["need_more_evidence"] is False
        assert result["query"] == ""
