from core.llm import llm as default_llm
from core.prompts import QUERY_REWRITER_PROMPT


def rewrite_query(query, chat_history=None, llm=None):
    if llm is None:
        llm = default_llm

    history_text = ""
    for message in (chat_history or [])[-4:]:
        history_text += f'{message["role"]}: {message["content"][:300]}\n'
    history_text = history_text.strip() or "None"

    prompt = QUERY_REWRITER_PROMPT.format(
        history_text=history_text,
        query=query,
    )
    try:
        rewritten_query = llm.invoke(prompt).content.strip()
    except Exception:
        return query

    return rewritten_query if rewritten_query else query
