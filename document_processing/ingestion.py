from langchain_core.documents import Document
from document_processing.text_splitter import split_documents

from document_processing.vector_store import (
    store_chunks,
    delete_document,
)

def ingest_documents(documents: list[Document]) -> int:

    if not documents:
        raise ValueError("No documents to ingest.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No valid chunks were generated.")

    # Get unique document sources.
    sources = {
        chunk.metadata["source"]
        for chunk in chunks
        if "source" in chunk.metadata
    }

    if not sources:
        raise ValueError("Document sources are missing.")

    # Remove old chunks before storing updated documents.
    for source in sources:
        delete_document(source)

    return store_chunks(chunks)
