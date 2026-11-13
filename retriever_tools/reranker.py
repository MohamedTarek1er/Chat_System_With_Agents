from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(query, documents, top_k=5):

    if not documents:
        return []

    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = model.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    results = [
        document
        for document, score in ranked_documents[:top_k]
    ]

    return results
