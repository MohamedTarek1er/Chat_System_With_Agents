import hashlib
import os
from typing import List, Optional, TypedDict
from langchain.agents import AgentState


class AnalystState(AgentState):
    evidence: list
    followup_query: str | None

class EvidenceItem(TypedDict):
    text: str
    doc_name: str
    page: Optional[int]      # 1-based, None if unknown
    chunk_id: str
    score: Optional[float]   # higher = more relevant, set by reranker only


class State(TypedDict, total=False):
    question: str
    chat_history: List[dict]
    evidence: List[EvidenceItem]
    analysis: str
    need_more_evidence: bool
    followup_query: str
    loop_count: int
    evidence_complete: bool   # False only if loop cap hit while still missing info
    final_answer: str


def to_evidence_item(doc) -> EvidenceItem:
    meta = doc.metadata or {}
    source = meta.get("document_name") or meta.get("source") or "unknown"

    page = meta.get("page_number")
    if page is None and isinstance(meta.get("page"), int):
        page = meta["page"] + 1          # legacy 0-indexed loaders

    chunk_id = (
        meta.get("chunk_id")
        or getattr(doc, "id", None)
        or hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
    )

    return {
        "text": doc.page_content,
        "doc_name": os.path.basename(str(source)),
        "page": page,
        "chunk_id": str(chunk_id),
        "score": None,
    }