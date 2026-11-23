from functools import lru_cache
from langchain_openai import ChatOpenAI
from core.config import LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

 
@lru_cache(maxsize=4)
def get_llm(temperature: float | None = LLM_TEMPERATURE) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing OPENAI_API_KEY. Create a .env file in the project root with OPENAI_API_KEY=your_key."
        )
    extra = {}
    if temperature is not None:
        extra["temperature"] = temperature
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        **extra
    )


def __getattr__(name: str):
    if name == "llm":
        return get_llm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
