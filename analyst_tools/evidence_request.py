import json

def evidence_request(llm, question, evidence):

    context = "\n\n".join(
        str(item) for item in evidence
    )

    prompt = f"""
    You are an evidence analysis assistant.

    User Question:
    {question}

    Available Evidence:
    {context}

    Check whether the available evidence is sufficient
    to answer the user's question.

    If sufficient:
        Return:
        {{
            "need_more_evidence": false,
            "query": ""
        }}

    If information is missing:
        Identify exactly what is missing and create
        a specific search query to retrieve it.

        Return:
        {{
            "need_more_evidence": true,
            "query": "specific search query"
        }}

    Do not invent information.
    Return valid JSON only.
    """

    response = llm.invoke(prompt)

    result = json.loads(response.content)
    return result
