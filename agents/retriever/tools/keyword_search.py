import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from ingestion.vector_store import get_vector_store
from graph.state import to_evidence_item


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_bm25_retriever(k=10):
    vector_store = get_vector_store()

    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        stored_data["documents"],
        stored_data["metadatas"]
    ):
        if content.strip():
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata or {}
                )
            )

    if not documents:
        raise ValueError("No documents found in the vector store.")

    retriever = BM25Retriever.from_documents(
        documents,
        k=k,
        preprocess_func=tokenize
    )

    return retriever


def keyword_search(query, retriever):
    results = retriever.invoke(query)
    return [to_evidence_item(result) for result in results]
