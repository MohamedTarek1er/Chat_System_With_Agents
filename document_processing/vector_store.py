import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store() -> Chroma:

    return Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

def generate_chunk_id(chunk: Document) -> str:

    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page_number", 0)
    start_index = chunk.metadata.get("start_index", 0)

    unique_text = (
        f"{source}|{page}|{start_index}|{chunk.page_content}"
    )

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


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

    # Keep only metadata types supported by Chroma.
    for chunk, chunk_id in zip(chunks, chunk_ids):

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
