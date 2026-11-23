import logging

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse
from graph.runner import run_pipeline

logger = logging.getLogger("api.query")
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    history = [message.model_dump() for message in request.chat_history]

    try:
        return run_pipeline(request.question, chat_history=history)
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=502,
            detail="The pipeline failed to produce an answer. Please try again.",
        )