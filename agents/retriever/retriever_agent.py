from agents.retriever.tools.query_rewriter import rewrite_query
from agents.retriever.tools.semantic_search import semantic_search
from agents.retriever.tools.keyword_search import keyword_search
from agents.retriever.tools.metadata_filter import filter_by_metadata
from agents.retriever.tools.reranker import rerank_documents
from agents.retriever.tools.context_selector import select_context
from core.config import RETRIEVER_CANDIDATES, RERANK_TOP_K, MAX_CHARS


def retrieve(query, chat_history=None, llm=None, bm25=None,
             filters=None, rewrite=True):
    search_query = rewrite_query(query, chat_history, llm) if rewrite else query

    candidates = semantic_search(search_query, k=RETRIEVER_CANDIDATES)

    if bm25 is not None:
        candidates += keyword_search(search_query, bm25)

    unique = {}
    for item in candidates:
        unique.setdefault(item["chunk_id"], item)
    candidates = list(unique.values())

    candidates = filter_by_metadata(candidates, filters)

    if not candidates:
        return []

    ranked = rerank_documents(search_query, candidates, top_k=RERANK_TOP_K)

    return select_context(ranked, max_chars=MAX_CHARS)
