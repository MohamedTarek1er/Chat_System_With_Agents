from functools import lru_cache
from langchain_openai import ChatOpenAI
from core.config import OPENAI_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing OPENAI_API_KEY. Create a .env file in the project root with OPENAI_API_KEY=your_key."
        )
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=OPENROUTER_MODEL,
        temperature=temperature,
    )


def __getattr__(name: str):
    if name == "llm":
        return get_llm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
