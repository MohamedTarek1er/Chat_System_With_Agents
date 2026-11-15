# Pipeline contract (RAG-Step 1)

Entry point: run_pipeline(question, chat_history=None) -> dict
  returns {"answer": str, "sources": list, "evidence_complete": bool}

Evidence item (the only evidence format after the retriever):
  {"text": "...", "doc_name": "paper.pdf", "page": 4, "chunk_id": "a1b2c3d4", "score": None}

Rules:
1. Only the retriever touches LangChain Documents. Convert with to_evidence_item().
2. Pages are 1-based. Unknown page is None, never 0.
3. Ingestion saves metadata keys: doc_name, page_number, chunk_id.
4. score is set only by the reranker (higher = better). Never store Chroma distances.
5. Nodes return only the State fields they change. Do not rename fields without telling both owners.
6. Loop cap hit while still missing info -> evidence_complete = False.