import io
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from api.routers import ocr

app = FastAPI()
app.include_router(ocr.router)
client = TestClient(app)


def _make_png_bytes(text: str | None = None) -> bytes:
    img = Image.new("RGB", (500, 120), color="white")
    if text:
        draw = ImageDraw.Draw(img)
        draw.text((30, 45), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_endpoint_extracts_text():
    img_bytes = _make_png_bytes("HELLO OCR 2026")
    response = client.post(
        "/ocr",
        files={"image": ("sample.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.png"
    assert "HELLO" in body["text"].upper()
    assert "2026" in body["text"]


def test_ocr_endpoint_blank_image():
    img_bytes = _make_png_bytes()
    response = client.post(
        "/ocr",
        files={"image": ("blank.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["text"].strip() == ""


def test_ocr_endpoint_rejects_unsupported_extension():
    response = client.post(
        "/ocr",
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_ocr_endpoint_rejects_empty_file():
    response = client.post(
        "/ocr",
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_ocr_endpoint_rejects_unknown_engine():
    img_bytes = _make_png_bytes("TEST")
    response = client.post(
        "/ocr",
        files={"image": ("sample.png", img_bytes, "image/png")},
        data={"engine": "nonexistent_engine"},
    )
    assert response.status_code == 422
