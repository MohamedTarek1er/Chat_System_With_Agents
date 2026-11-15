from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(query, documents, top_k=5):

    if not documents:
        return []

    pairs = [(query, item["text"]) for item in documents]
    scores = model.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [{**item, "score": float(score)} for item, score in ranked_documents[:top_k]]