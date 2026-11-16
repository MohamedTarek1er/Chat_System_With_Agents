from core.llm import llm
from core.prompts import DOCUMENT_COMPARISON_PROMPT


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

    prompt = DOCUMENT_COMPARISON_PROMPT.format(
        query=query,
        context=context,
    )

    response = llm.invoke(prompt)

    return response.content
