from retriever_tools.query_rewriter import rewrite_query
from retriever_tools.semantic_search import semantic_search
from retriever_tools.keyword_search import (
    build_bm25_retriever,
    keyword_search
)
from retriever_tools.metadata_filter import filter_by_metadata
from retriever_tools.reranker import rerank_documents
from retriever_tools.context_selector import (
    select_context,
    format_context
)


# Build BM25 index once
bm25_retriever = build_bm25_retriever()


# User question
user_query = "How does deep learning work?"


# 1. Query Rewriter
rewritten_query = rewrite_query(user_query)

print("\nRewritten Query:")
print(rewritten_query)


# 2. Semantic Search
semantic_results = semantic_search(rewritten_query)


# 3. Keyword Search
keyword_results = keyword_search(
    rewritten_query,
    bm25_retriever
)


# Combine results from both searches
all_results = semantic_results + keyword_results


# 4. Metadata Filter
filtered_results = filter_by_metadata(
    all_results,
    {"file_type": ".pdf"}
)


# 5. Reranker
ranked_results = rerank_documents(
    rewritten_query,
    filtered_results,
    top_k=5
)


# 6. Context Selector
selected_documents = select_context(
    ranked_results,
    max_chars=12000
)


# Prepare final context
final_context = format_context(selected_documents)


print("\nFinal Retrieved Context:")
print(final_context)
