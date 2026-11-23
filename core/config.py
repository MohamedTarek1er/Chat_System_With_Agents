import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# LLM Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

_temp = os.getenv("LLM_TEMPERATURE","").strip()
LLM_TEMPERATURE = float(_temp) if _temp else None

# Model Names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Orchestrator & Loop Settings
MAX_LOOPS = 3

# Retriever Settings
RETRIEVER_CANDIDATES = 10
RERANK_TOP_K = 8
MAX_CHARS = 12000

# Vector Store Settings
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTION_NAME = "knowledge_base"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
