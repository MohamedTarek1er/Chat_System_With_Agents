from langchain.agents import create_agent        # older: from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from core.prompts import ANALYST_SYSTEM_PROMPT
from graph.state import State
from graph.state import AnalystState
from .tools.calculator import calculate_tool
from .tools.data_analysis import data_analysis
from .tools.document_comparison import compare_documents
from .tools.evidence_request import format_evidence, request_more_evidence
from .tools.table_extractor import extract_tables_tool


RECURSION_LIMIT = 12   # max steps inside ONE analysis (not the retriever<->analyst loop cap)


def make_analyst_node(llm):
    """llm: e.g. get_llm(temperature=0), built once in graph/runner.py."""
    agent = create_agent(
        model=llm,
        tools=[calculate_tool, extract_tables_tool, data_analysis, 
            compare_documents, request_more_evidence],
        system_prompt=ANALYST_SYSTEM_PROMPT,
        state_schema=AnalystState,
    )

    def analyst_node(state: State) -> dict:
        evidence = state.get("evidence", [])
        user_msg = (f"Question: {state['question']}\n\n"
                    f"Retrieved evidence:\n{format_evidence(evidence)}")
        try:
            out = agent.invoke(
                {"messages": [("user", user_msg)],
                "evidence": evidence,
                "followup_query": None},
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except GraphRecursionError:
            return {"analysis": "Analysis step limit reached; answering with available evidence.",
                    "need_more_evidence": False, "followup_query": ""}

        followup = out.get("followup_query") or ""
        need_more = bool(followup)
        content = out["messages"][-1].content
        return {
            "analysis": content if isinstance(content, str) else str(content),
            "need_more_evidence": need_more,
            "followup_query": followup,
        }

    return analyst_node