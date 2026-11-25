from pathlib import Path
import pytest
from core.ocr.factory import OCRFactory

SAMPLES_DIR = Path(__file__).parent / "ocr_samples"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def get_sample_images():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


@pytest.mark.parametrize("img_path", get_sample_images(), ids=lambda p: p.name)
def test_real_sample_images(img_path):
    engine = OCRFactory.create("rapidocr")
    text = engine.extract_text(img_path)
    print(f"\n[{img_path.name}]\n{text}")
    assert text.strip() != ""