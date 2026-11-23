import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

# These imports pull in the pipeline, so the heavy models load once,
# when the server starts, not on the first request.
from api.routers import documents, query
from api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    documents.UPLOAD_DIR.mkdir(exist_ok=True)
    logger.info("API ready")
    yield


app = FastAPI(title="Multi-Agent RAG API", version="0.1.0", lifespan=lifespan)

app.include_router(query.router)
app.include_router(documents.router)
# Later: voice, ocr and report routers get one include_router line each.

# Later: mount the HTML front end here, after every router:
# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return {"status": "ok"}