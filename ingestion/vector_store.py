import hashlib
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from core.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def generate_chunk_id(chunk: Document) -> str:
    if chunk.metadata.get("chunk_id"):
        return str(chunk.metadata["chunk_id"])

    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page_number", 0)
    start_index = chunk.metadata.get("start_index", 0)

    unique_text = f"{source}|{page}|{start_index}|{chunk.page_content}"

    return hashlib.md5(unique_text.encode("utf-8")).hexdigest()[:8]


def store_chunks(chunks: list[Document]) -> int:
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No valid chunks to store.")

    vector_store = get_vector_store()

    chunk_ids = [
        generate_chunk_id(chunk)
        for chunk in chunks
    ]

    # Keep only metadata types supported by Chroma and ensure doc_name is populated.
    for chunk, chunk_id in zip(chunks, chunk_ids):
        # if not chunk.metadata.get("doc_name"):
        #     chunk.metadata["doc_name"] = os.path.basename(
        #         str(chunk.metadata.get("document_name") or chunk.metadata.get("source", "unknown"))
        #     )
        chunk.metadata = {
            key: value
            for key, value in chunk.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        chunk.metadata["chunk_id"] = chunk_id

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return len(chunks)


def delete_document(source: str) -> None:
    vector_store = get_vector_store()
    vector_store.delete(
        where={"source": source}
    )

def clear_vector_store() -> None:
    vector_store = get_vector_store()
    try:
        vector_store.reset_collection()
    except Exception:
        try:
            stored = vector_store.get()
            if stored and "ids" in stored and stored["ids"]:
                vector_store.delete(ids=stored["ids"])
        except Exception:
            pass


def search_documents(
    query: str,
    k: int = 4
) -> list[Document]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query.strip(),
        k=k,
    )
