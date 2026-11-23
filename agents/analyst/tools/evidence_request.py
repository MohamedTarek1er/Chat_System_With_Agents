from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


def format_evidence(evidence) -> str:
    """Render EvidenceItem dicts as numbered, cited excerpts for the LLM."""
    lines = []
    for i, item in enumerate(evidence, start=1):
        doc = item.get("doc_name", "unknown")
        page = item.get("page")
        page_str = f", p. {page}" if page is not None else ""
        lines.append(f"[{i}] ({doc}{page_str})\n{item.get('text', '').strip()}")
    return "\n\n".join(lines)


@tool
def request_more_evidence(
    query: str,
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Call this when the retrieved evidence is NOT enough to answer the question.

    Args:
        query: A specific, self-contained search query for the missing information,
            e.g. "computational complexity of the ResNet model". Never a vague request.
        reason: One sentence on what is missing and why it is needed.
    """
    query = query.strip()
    if not query:
        return Command(update={"messages": [ToolMessage(
            "Error: 'query' must describe the missing information.",
            tool_call_id=tool_call_id)]})

    return Command(update={
        "followup_query": query,
        "messages": [ToolMessage(
            f"Request recorded: the retriever will search for '{query}'. "
            "Stop calling tools and briefly summarise what the evidence already establishes.",
            tool_call_id=tool_call_id)],
    })
