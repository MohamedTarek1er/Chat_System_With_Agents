from pathlib import Path

import pdfplumber
from docx import Document

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
