"""core/speech.py - speech-to-text with faster-whisper (runs locally, no API key)."""

import io
import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "120"))

_model = None
_load_lock = threading.Lock()
_run_lock = threading.Lock() 


class SpeechError(Exception):
    """The audio could not be turned into text (unreadable, too long, or silent)."""


def get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading Whisper model '%s' (%s, %s)...",
                    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
                )
                _model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                logger.info("Whisper model ready")
    return _model


def warm_up():
    """Load the model in the background at server start."""
    try:
        get_model()
    except Exception:
        logger.exception("Whisper warm-up failed; voice queries will fail until fixed")


def transcribe(audio_bytes, language=None, translate=False):
    """Return {"text", "language", "duration"} for the given audio bytes."""
    model = get_model()

    with _run_lock:
        try:
            segments, info = model.transcribe(
                io.BytesIO(audio_bytes),
                language=language,
                task="translate" if translate else "transcribe",
                beam_size=5,
                vad_filter=True,  # skips silence, which stops Whisper inventing text
            )
        except Exception as error:
            raise SpeechError(
                "The audio could not be processed (unreadable file or bad language code)."
            ) from error

        if info.duration > MAX_AUDIO_SECONDS:
            raise SpeechError(
                f"Audio is too long ({info.duration:.0f}s). Maximum is {MAX_AUDIO_SECONDS}s."
            )

        text = " ".join(segment.text.strip() for segment in segments).strip()

    if not text:
        raise SpeechError("No speech was detected in the audio.")

    return {"text": text, "language": info.language, "duration": round(info.duration, 1)}