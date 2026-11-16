"""
ingestion – Document preparation and vector store pipeline.
"""

from ingestion.loader import load_document
from ingestion.splitter import split_documents
from ingestion.vector_store import (
    get_vector_store,
    store_chunks,
    delete_document,
    search_documents,
)
from ingestion.ingestion import ingest_documents

__all__ = [
    "load_document",
    "split_documents",
    "get_vector_store",
    "store_chunks",
    "delete_document",
    "search_documents",
    "ingest_documents",
]
