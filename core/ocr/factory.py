from functools import lru_cache
from core.config import OCR_ENGINE
from core.ocr.base import BaseOCR


class OCRFactory:
    @staticmethod
    @lru_cache(maxsize=4)
    def create(engine_name: str = OCR_ENGINE) -> BaseOCR:
        name = engine_name.lower().strip()

        if name == "rapidocr":
            from core.ocr.rapid_ocr import RapidOCREngine
            return RapidOCREngine()

        # if name == "tesseract":
        #     from ocr.tesseract_ocr import TesseractOCREngine
        #     return TesseractOCREngine()

        raise ValueError(f"Unsupported OCR engine: '{engine_name}'")
