from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}

def load_document(file_path: str):

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    else:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    documents = loader.load()

    for document in documents:

        document.metadata["source"] = str(path.resolve())
        document.metadata["document_name"] = path.name
        document.metadata["file_type"] = extension

        # PDF page numbers start from 0 in PyPDFLoader.
        if "page" in document.metadata:
            document.metadata["page_number"] = (
                document.metadata["page"] + 1
            )

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]
