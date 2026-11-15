from document_processing.vector_store import search_documents
from state import to_evidence_item
def semantic_search(query, k=4):

    results = search_documents(query, k)
    return [to_evidence_item(result) for result in results]
