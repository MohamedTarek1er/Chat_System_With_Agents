from llm_setup import llm

def compare_documents(query, documents):

    if not documents:
        return "No evidence was provided for comparison."

    grouped_evidence = {}
    for document in documents:

        metadata = document.metadata
        source = metadata.get(
            "document_name",
            metadata.get("source", "Unknown document")
        )

        page = metadata.get("page_number", "Unknown")

        if source not in grouped_evidence:
            grouped_evidence[source] = []

        grouped_evidence[source].append(
            f"Page {page}:\n{document.page_content}"
        )

    context = ""

    for source, contents in grouped_evidence.items():

        context += f"\nDocument: {source}\n"
        context += "\n".join(contents)

    prompt = f"""
    You are a document comparison assistant.

    Compare the provided documents based only on their evidence.

    User request:
    {query}

    Document evidence:
    {context}

    Instructions:
    - Identify similarities and differences.
    - Compare methodologies, results, and conclusions when available.
    - Mention advantages and disadvantages only when supported.
    - Do not invent missing information.
    - If information is unavailable, state that clearly.
    - Include document names and page numbers as references.
    """

    response = llm.invoke(prompt)

    return response.content
