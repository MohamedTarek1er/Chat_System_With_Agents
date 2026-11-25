# tests/test_ocr.py
import io
import pytest
from PIL import Image, ImageDraw

from core.ocr.base import BaseOCR
from core.ocr.factory import OCRFactory


@pytest.fixture
def sample_image_path(tmp_path):
    """Generate a clean PNG with text on the fly using Pillow."""
    img = Image.new("RGB", (500, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 45), "HELLO OCR 2026", fill="black")

    path = tmp_path / "sample.png"
    img.save(path)
    return path


@pytest.fixture
def blank_image_bytes():
    """Generate a blank white PNG as bytes (should extract empty text)."""
    img = Image.new("RGB", (200, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- 1. Factory Tests ---

def test_factory_returns_base_ocr_instance():
    engine = OCRFactory.create("rapidocr")
    assert isinstance(engine, BaseOCR)


def test_factory_caches_engine_instance():
    e1 = OCRFactory.create("rapidocr")
    e2 = OCRFactory.create("rapidocr")
    assert e1 is e2


def test_factory_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        OCRFactory.create("nonexistent_engine")


# --- 2. Extraction Tests ---

def test_extract_text_from_file_path(sample_image_path):
    engine = OCRFactory.create("rapidocr")
    text = engine.extract_text(sample_image_path)

    assert "HELLO" in text.upper()
    assert "2026" in text


def test_extract_text_from_bytes(sample_image_path):
    engine = OCRFactory.create("rapidocr")
    raw_bytes = sample_image_path.read_bytes()
    text = engine.extract_text(raw_bytes)

    assert "HELLO" in text.upper()


def test_extract_text_from_blank_image_returns_empty_string(blank_image_bytes):
    engine = OCRFactory.create("rapidocr")
    text = engine.extract_text(blank_image_bytes)

    assert text.strip() == ""