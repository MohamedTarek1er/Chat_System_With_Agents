from langchain_openai import ChatOpenAI
from core.config import OPENAI_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=OPENROUTER_MODEL,
)
