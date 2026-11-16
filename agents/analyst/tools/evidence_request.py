import json
import re

from core.prompts import EVIDENCE_REQUEST_PROMPT


def _format_evidence(evidence):
    """Format evidence items cleanly for the LLM prompt.

    Handles EvidenceItem dicts (text, doc_name, page, chunk_id, score)
    and falls back gracefully for anything unexpected.
    """
    lines = []
    for i, item in enumerate(evidence, start=1):
        if isinstance(item, dict):
            text = item.get("text", "")
            doc = item.get("doc_name", "unknown")
            page = item.get("page")
            page_str = f", p. {page}" if page is not None else ""
            lines.append(f"[{i}] ({doc}{page_str})\n{text.strip()}")
        elif hasattr(item, "page_content"):
            meta = item.metadata or {}
            doc = meta.get("doc_name", meta.get("source", "unknown"))
            page = meta.get("page_number", meta.get("page"))
            page_str = f", p. {page}" if page is not None else ""
            lines.append(f"[{i}] ({doc}{page_str})\n{item.page_content.strip()}")
        else:
            lines.append(f"[{i}]\n{str(item).strip()}")
    return "\n\n".join(lines)


def _parse_json_safe(text):
    """Parse JSON from LLM output, handling code fences and garbage.

    Returns a dict with need_more_evidence (bool) and query (str).
    Falls back to {need_more_evidence: False, query: ""} on any failure.
    """
    fallback = {"need_more_evidence": False, "query": ""}

    text = text.strip()

    # Strip markdown code fences if present: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback

    if not isinstance(result, dict):
        return fallback

    if "need_more_evidence" not in result:
        return fallback

    return {
        "need_more_evidence": bool(result["need_more_evidence"]),
        "query": str(result.get("query", "")),
    }


def evidence_request(llm, question, evidence):
    """Ask the LLM whether the evidence is sufficient for the question.

    Returns {"need_more_evidence": bool, "query": str}.
    """
    context = _format_evidence(evidence)

    prompt = EVIDENCE_REQUEST_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(prompt)

    return _parse_json_safe(response.content)
