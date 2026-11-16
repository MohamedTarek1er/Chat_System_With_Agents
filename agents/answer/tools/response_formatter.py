from core.llm import llm
from core.prompts import RESPONSE_FORMATTER_PROMPT


def generate_response(question, analysis, context):
    prompt = RESPONSE_FORMATTER_PROMPT.format(
        question=question,
        analysis=analysis,
        context=context,
    )

    response = llm.invoke(prompt)

    return response.content
