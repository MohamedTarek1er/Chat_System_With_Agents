from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.llm import get_llm
from core.prompts import DOCUMENT_COMPARISON_PROMPT


@tool
def compare_documents(
    query: str,
    evidence: Annotated[list, InjectedState("evidence")],
) -> str:
    """Compare what the different retrieved documents say about `query`:
    similarities, differences, methods, results and conclusions."""
    if not evidence:
        return "No evidence was provided for comparison."

    grouped: dict[str, list[str]] = {}
    for item in evidence:
        page = item.get("page")
        grouped.setdefault(item["doc_name"], []).append(
            f"Page {page if page is not None else '?'}:\n{item['text']}"
        )
    context = "\n\n".join(
        f"Document: {doc}\n" + "\n".join(chunks) for doc, chunks in grouped.items()
    )

    prompt = DOCUMENT_COMPARISON_PROMPT.format(query=query, context=context)
    return get_llm(temperature=0).invoke(prompt).content
