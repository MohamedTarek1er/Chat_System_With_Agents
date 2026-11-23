import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import DocumentList, UploadResponse , DeleteResponse
from ingestion.ingestion import ingest_documents
from ingestion.loader import load_document
from ingestion.vector_store import get_vector_store

logger = logging.getLogger("api.documents")
router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/documents", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (max 20 MB).")

    # Same filename -> same path, so re-uploading replaces the old chunks.
    UPLOAD_DIR.mkdir(exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(data)

    try:
        documents = load_document(str(path))
    except ImportError as error:
        logger.exception("Missing dependency %s", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Server is missing a required package ({error}).",
        )
    except Exception:
        logger.exception("Could not read %s", filename)
        raise HTTPException(
            status_code=422,
            detail="The file could not be read. It may be corrupted or password-protected.",
        )

    try:
        chunks = ingest_documents(documents)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=f"No readable text found in this file ({error}).",
        )
    except Exception:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail="Failed to process the document.")

    return {"document_name": filename, "chunks_stored": chunks}


@router.get("/documents", response_model=DocumentList)
def list_documents():
    data = get_vector_store().get(include=["metadatas"])
    names = set()
    for metadata in data.get("metadatas") or []:
        if not metadata:
            continue
        name = (
            metadata.get("doc_name")
            or metadata.get("document_name")
            or Path(str(metadata.get("source", ""))).name
        )
        if name:
            names.add(name)
    return {"documents": sorted(names)}


@router.delete("/documents/{document_name}", response_model=DeleteResponse)
def remove_document(document_name: str):
    store = get_vector_store()
    ids = store.get(where={"doc_name": document_name}, include=["metadatas"])["ids"]

    if not ids:
        raise HTTPException(
            status_code=404,
            detail=f"No document named '{document_name}'. Use GET /documents for the exact names.",
        )

    store.delete(ids=ids)

    # Also remove the saved copy so uploads/ matches the index.
    (UPLOAD_DIR / Path(document_name).name).unlink(missing_ok=True)

    return {"document_name": document_name, "chunks_deleted": len(ids)}
