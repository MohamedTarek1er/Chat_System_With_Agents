import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import TypeAdapter, ValidationError

from api.schemas import ChatMessage, VoiceQueryResponse
from core.speech import SpeechError, transcribe
from graph.runner import run_pipeline

logger = logging.getLogger("api.voice")
router = APIRouter(tags=["voice"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".aac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024
_history_adapter = TypeAdapter(list[ChatMessage])


@router.post("/voice-query", response_model=VoiceQueryResponse)
def voice_query(
    audio: UploadFile = File(...),
    chat_history: str = Form("[]"),
    language: Optional[str] = Form(None),
    translate: bool = Form(False),
):
    extension = Path(audio.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    language = (language or "").strip().lower() or None
    if language and not re.fullmatch(r"[a-z]{2,3}", language):
        raise HTTPException(status_code=422, detail="language must be a code like 'en' or 'ar'.")

    try:
        history = [m.model_dump() for m in _history_adapter.validate_json(chat_history)][-20:]
    except ValidationError:
        raise HTTPException(
            status_code=422,
            detail="chat_history must be a JSON list of {role, content} objects.",
        )

    data = audio.file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded audio is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is too large (max 25 MB).")

    try:
        speech = transcribe(data, language=language, translate=translate)
    except SpeechError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except ImportError as error:
        logger.exception("Speech dependency missing")
        raise HTTPException(status_code=500, detail=f"Server is missing a required package: {error}")
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Speech recognition failed.")

    try:
        result = run_pipeline(speech["text"], chat_history=history)
    except Exception:
        logger.exception("Pipeline failed for voice query")
        raise HTTPException(
            status_code=502,
            detail="The pipeline failed to produce an answer. Please try again.",
        )

    return {
        **result,
        "transcript": speech["text"],
        "language": speech["language"],
        "duration_seconds": speech["duration"],
    }