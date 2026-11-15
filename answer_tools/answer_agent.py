from answer_tools.source_formatter import format_sources
from answer_tools.response_formatter import generate_response
from answer_tools.citation_formatter import format_citations


def answer_question(question, analysis, evidence, evidence_complete=True):
    """Build the final answer from analysis + evidence.

    Args:
        question:          The user's original question.
        analysis:          Analyst output (str). May be empty.
        evidence:          List of EvidenceItem dicts or LangChain Documents.
        evidence_complete: False when the evidence-loop cap was hit while the
                           analyst still wanted more info.

    Returns:
        dict with keys "answer" (str) and "evidence_complete" (bool).
    """
    # Guard: no evidence at all
    if not evidence:
        return {
            "answer": "I couldn't find sufficient evidence to answer this question.",
            "evidence_complete": evidence_complete,
        }

    # Guard: empty / whitespace-only analysis
    if not analysis or not analysis.strip():
        analysis = "No analyst findings were produced for this question."

    formatted_data = format_sources(evidence)

    context = formatted_data["context"]
    sources = formatted_data["sources"]

    # If evidence is incomplete, tell the LLM so it can qualify its answer
    if not evidence_complete:
        analysis += (
            "\n\n[NOTE: The evidence-gathering loop reached its cap before all "
            "requested information could be retrieved. The answer may be based "
            "on incomplete evidence.]"
        )

    response = generate_response(
        question,
        analysis,
        context
    )

    final_answer = format_citations(
        response,
        sources
    )

    return {
        "answer": final_answer,
        "evidence_complete": evidence_complete,
    }
