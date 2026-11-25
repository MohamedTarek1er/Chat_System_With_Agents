import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas import OCRResponse
from core.config import OCR_ENGINE
from core.ocr.factory import OCRFactory

logger = logging.getLogger("api.ocr")
router = APIRouter(tags=["ocr"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@router.post("/ocr", response_model=OCRResponse)
async def extract_text_from_image(
    image: UploadFile = File(...),
    engine: Optional[str] = Form(None),
):
    filename = Path(image.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 20 MB).")

    engine_name = (engine or OCR_ENGINE).strip()

    try:
        ocr_engine = await asyncio.to_thread(OCRFactory.create, engine_name)
        text = await asyncio.to_thread(ocr_engine.extract_text, data)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except ImportError as error:
        logger.exception("OCR dependency missing")
        raise HTTPException(status_code=500, detail=f"Server is missing a required package: {error}")
    except Exception:
        logger.exception("OCR extraction failed for %s", filename)
        raise HTTPException(status_code=500, detail="OCR text extraction failed.")

    return {"filename": filename, "text": text}
