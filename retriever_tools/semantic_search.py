from document_processing.vector_store import search_documents

def semantic_search(query, k=4):

    results = search_documents(query, k)
    return results
