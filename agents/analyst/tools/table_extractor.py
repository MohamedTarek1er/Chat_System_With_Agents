from pathlib import Path
from langchain_core.tools import tool
from core.config import UPLOAD_DIR
import pdfplumber
from docx import Document


@tool
def extract_tables_tool(doc_name: str, page: int | None = None) -> str:
    """Extract tables from an uploaded PDF/DOCX by its document name (as shown in the
    evidence), optionally only from one page. Returns markdown-style rows."""
    path = Path(UPLOAD_DIR) / doc_name
    if not path.is_file():
        return f"Error: file '{doc_name}' not found in uploads directory."
    tables = extract_tables(path)
    if page is not None:
        tables = [t for t in tables if t["page"] == page]
    if not tables:
        return "No tables found."
    out = []
    for t in tables:
        rows = "\n".join(" | ".join(str(c or "") for c in r) for r in t["data"])
        out.append(f"[{t['source']} p.{t['page']} table {t['table_number']}]\n{rows}")
    return "\n\n".join(out)[:6000] 


def extract_pdf_tables(file_path):

    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            page_tables = page.extract_tables()
            for table_number, table in enumerate(page_tables, start=1):

                if not table:
                    continue

                tables.append({
                    "source": Path(file_path).name,
                    "page": page_number,
                    "table_number": table_number,
                    "data": table
                })

    return tables


def extract_docx_tables(file_path):

    document = Document(file_path)
    tables = []

    for table_number, table in enumerate(document.tables, start=1):

        rows = []
        for row in table.rows:

            row_data = [
                cell.text.strip()
                for cell in row.cells
            ]

            rows.append(row_data)

        tables.append({
            "source": Path(file_path).name,
            "page": None,
            "table_number": table_number,
            "data": rows
        })

    return tables


def extract_tables(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_tables(file_path)

    elif extension == ".docx":
        return extract_docx_tables(file_path)

    else:
        raise ValueError("Only PDF and DOCX files are supported.")
