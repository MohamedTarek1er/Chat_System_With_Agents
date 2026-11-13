from answer_tools.source_formatter import format_sources
from answer_tools.response_formatter import generate_response
from answer_tools.citation_formatter import format_citations

def answer_question(question, analysis, evidence):

    if not evidence:
        return "I couldn't find sufficient evidence to answer this question."

    formatted_data = format_sources(evidence)

    context = formatted_data["context"]
    sources = formatted_data["sources"]

    response = generate_response(
        question,
        analysis,
        context
    )

    final_answer = format_citations(
        response,
        sources
    )

    return final_answer
