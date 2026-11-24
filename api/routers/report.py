import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from agents.report.report_agent import generate_report
from api.schemas import ChatMessage, Source
from core.config import BASE_DIR
from graph.runner import run_pipeline


logger = logging.getLogger("api.report")
router = APIRouter(tags=["report"])
REPORT_DIR = BASE_DIR / "reports"


class ReportRequest(BaseModel):

    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field("", max_length=50000)
    analysis: str = Field("", max_length=50000)

    sources: list[Source] = Field(default_factory=list)
    style_request: str = Field("", max_length=1000)

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20
    )

@router.post("/report")
def create_report(request: ReportRequest):

    answer = request.answer
    analysis = request.analysis

    sources = [
        source.model_dump()
        for source in request.sources
    ]

    if not answer.strip():

        history = [
            message.model_dump()
            for message in request.chat_history
        ]

        try:
            result = run_pipeline(
                request.question,
                chat_history=history
            )

            answer = result["answer"]
            analysis = result.get("analysis", "")
            sources = result.get("sources", [])

        except Exception:
            logger.exception("Pipeline failed")

            raise HTTPException(
                status_code=500,
                detail="Could not generate the answer."
            )

    try:
        path = generate_report(
            question=request.question,
            answer=answer,
            analysis=analysis,
            sources=sources,
            style_request=request.style_request,
            output_dir=REPORT_DIR
        )

    except Exception:
        logger.exception("Report generation failed")

        raise HTTPException(
            status_code=500,
            detail="Could not generate the PDF report."
        )

    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename="report.pdf",
        background=BackgroundTask(
            path.unlink,
            missing_ok=True
        )
    )
