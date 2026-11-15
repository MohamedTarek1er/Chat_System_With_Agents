from llm_setup import llm

def rewrite_query(query,chat_history=None,llm=None):

    if llm is None:
        from llm_setup import llm as default_llm
        llm = default_llm

    history_text=""
    for message in (chat_history or [])[-4:]:
       history_text += f'{message["role"]}: {message["content"][:300]}\n'
    history_text = history_text.strip() or None  

    prompt = f"""
    You are a query rewriting assistant.

    Rewrite the user's question to make it clearer
    and more suitable for searching documents.

    Rules:
    - Keep the original meaning.
    - Do not answer the question.
    - Use the conversation history only to replace pronouns or vague references (e.g. "it", "its", "that method").
    - Do not add information that is not in the question or history.
    - Return only the rewritten question.
    conversation history:
    {history_text}

    User question:
    {query}
    """
    try:
        rewritten_query = llm.invoke(prompt).content.strip()
    except Exception: 
        return query   

    return rewritten_query if rewritten_query else query
