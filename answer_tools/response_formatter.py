
from llm_setup import llm


def generate_response(question, analysis, context):

    prompt = f"""
You are the Answer Agent in a multi-agent document chat system.

Your responsibility is to generate the final answer for the user.

USER QUESTION:
{question}

ANALYST RESULTS:
{analysis}

AVAILABLE DOCUMENT EVIDENCE:
{context}

INSTRUCTIONS:

1. Answer the user's original question directly.
2. Use the Analyst results to explain the findings.
3. Use the provided evidence to support your answer.
4. Do not invent information or unsupported conclusions.
5. Do not introduce facts that are not supported by the evidence.
6. Include source IDs as citations, such as [S1] or [S2].
7. Only use source IDs that actually exist in the provided evidence.
8. If information is missing, clearly mention that.
9. Organize the answer using suitable formatting.

RESPONSE FORMATTING:

- Use headings when they improve readability.
- Use bullet points for multiple findings.
- Use tables when comparing multiple items.
- Explain calculations in understandable terms.
- Include citations next to the claims they support.
- Avoid unnecessary repetition.

Return only the final answer.
"""

    response = llm.invoke(prompt)

    return response.content
