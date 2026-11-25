from pathlib import Path
from rapidocr import RapidOCR
from core.ocr.base import BaseOCR


class RapidOCREngine(BaseOCR):
    def __init__(self):
        self._engine = RapidOCR()  # Models loaded once

    def extract_text(self, image_source: str | Path | bytes) -> str:
        result = self._engine(image_source)
        if not result or not result.txts:
            return ""
        return "\n".join(result.txts)
