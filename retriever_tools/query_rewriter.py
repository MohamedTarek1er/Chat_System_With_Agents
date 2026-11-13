from llm_setup import llm

def rewrite_query(query):

    prompt = f"""
    You are a query rewriting assistant.

    Rewrite the user's question to make it clearer
    and more suitable for searching documents.

    Rules:
    - Keep the original meaning.
    - Do not answer the question.
    - Do not add information that is not provided.
    - Return only the rewritten question.

    User question:
    {query}
    """

    response = llm.invoke(prompt)
    rewritten_query = response.content.strip()

    return rewritten_query if rewritten_query else query
