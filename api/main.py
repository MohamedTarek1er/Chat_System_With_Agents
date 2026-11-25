import logging
from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI

# These imports pull in the pipeline, so the heavy models load once,
# when the server starts, not on the first request.
from api.routers import documents, query, voice, report, ocr
from api.schemas import HealthResponse
from core.speech import warm_up

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    documents.UPLOAD_DIR.mkdir(exist_ok=True)
    threading.Thread(target=warm_up, daemon=True).start()
    logger.info("API ready")
    yield


app = FastAPI(title="Multi-Agent RAG API", version="0.1.0", lifespan=lifespan)

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(voice.router)
app.include_router(report.router)
app.include_router(ocr.router)

# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return {"status": "ok"}