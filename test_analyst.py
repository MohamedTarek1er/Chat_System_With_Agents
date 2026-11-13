import os
import tempfile

from pathlib import Path
from docx import Document
from langchain_core.documents import Document as LCDocument

from llm_setup import llm

from analyst_tools.calculator import calculate
from analyst_tools.table_extractor import extract_tables
from analyst_tools.document_comparison import compare_documents
from analyst_tools.data_analysis import data_analysis
from analyst_tools.evidence_request import evidence_request


# ==========================================
# Sample Evidence
# ==========================================

evidence = [
    LCDocument(
        page_content="""
        Model A achieved 92% accuracy and 90% precision.
        Training time was 30 minutes.
        """,
        metadata={
            "document_name": "paper1.pdf",
            "page_number": 3
        }
    ),

    LCDocument(
        page_content="""
        Model B achieved 87% accuracy and 85% precision.
        Training time was 20 minutes.
        """,
        metadata={
            "document_name": "paper2.pdf",
            "page_number": 5
        }
    ),

    LCDocument(
        page_content="""
        Model C achieved 95% accuracy and 93% precision.
        Training time was 25 minutes.
        """,
        metadata={
            "document_name": "paper3.pdf",
            "page_number": 7
        }
    )
]


# ==========================================
# 1. Test Calculator
# ==========================================

def test_calculator():

    print("\n========== CALCULATOR TEST ==========")

    print("Addition:", calculate("add", [10, 20, 30]))
    print("Subtraction:", calculate("subtract", [50, 10, 5]))
    print("Multiplication:", calculate("multiply", [2, 3, 4]))
    print("Division:", calculate("divide", [100, 2, 5]))
    print("Average:", calculate("average", [80, 90, 100]))
    print("Median:", calculate("median", [10, 20, 30]))
    print("Percentage:", calculate("percentage", [25, 100]))
    print("Ratio:", calculate("ratio", [10, 20]))

    print("Calculator test passed!")


# ==========================================
# 2. Test Table Extractor
# ==========================================

def test_table_extractor():

    print("\n========== TABLE EXTRACTOR TEST ==========")

    # Create a temporary DOCX containing a table
    with tempfile.TemporaryDirectory() as temp_dir:

        file_path = Path(temp_dir) / "sample.docx"

        document = Document()

        table = document.add_table(
            rows=1,
            cols=3
        )

        headers = ["Model", "Accuracy", "Precision"]

        for i, header in enumerate(headers):
            table.rows[0].cells[i].text = header

        data = [
            ["Model A", "92%", "90%"],
            ["Model B", "87%", "85%"],
            ["Model C", "95%", "93%"]
        ]

        for row_data in data:

            cells = table.add_row().cells

            for i, value in enumerate(row_data):
                cells[i].text = value

        document.save(file_path)

        # Extract tables
        result = extract_tables(str(file_path))

        print("Extracted tables:")

        for table in result:
            print(table)

        assert len(result) > 0

    print("Table Extractor test passed!")


# ==========================================
# 3. Test Document Comparison
# ==========================================

def test_document_comparison():

    print("\n========== DOCUMENT COMPARISON TEST ==========")

    question = """
    Compare the three models based on accuracy,
    precision, and training time.
    Identify their similarities and differences.
    """

    result = compare_documents(
        question,
        evidence
    )

    print(result)

    assert isinstance(result, str)
    assert len(result) > 0

    print("Document Comparison test passed!")


# ==========================================
# 4. Test Data Analysis
# ==========================================

def test_data_analysis():

    print("\n========== DATA ANALYSIS TEST ==========")

    question = """
    Compare the three models based on accuracy,
    precision, and training time.
    Identify the best performing model.
    """

    result = data_analysis(
        llm,
        question,
        evidence
    )

    print("\nAnalysis Result:")

    print(result)

    assert result["success"] is True

    print("\nData Analysis test passed!")


# ==========================================
# 5. Test Evidence Request
# ==========================================

def test_evidence_request():

    print("\n========== EVIDENCE REQUEST TEST ==========")

    question = """
    Compare the three models based on accuracy,
    precision, training time, and memory usage.
    """

    result = evidence_request(
        llm,
        question,
        evidence
    )

    print("\nEvidence Request Result:")

    print(result)

    assert "need_more_evidence" in result
    assert "query" in result

    if result["need_more_evidence"]:

        print("\nAdditional evidence required:")
        print(result["query"])

    else:

        print("\nEvidence is sufficient.")

    print("Evidence Request test passed!")


# ==========================================
# 6. Run All Tests
# ==========================================

if __name__ == "__main__":

    tests = [
        ("Calculator", test_calculator),
        ("Table Extractor", test_table_extractor),
        ("Document Comparison", test_document_comparison),
        ("Data Analysis", test_data_analysis),
        ("Evidence Request", test_evidence_request)
    ]

    passed = 0
    failed = 0

    for name, test_function in tests:

        try:

            test_function()
            passed += 1

        except Exception as e:

            failed += 1

            print(f"\n{name} FAILED!")
            print("Error:", e)

    print("\n====================================")
    print("          TEST SUMMARY")
    print("====================================")

    print("Passed:", passed)
    print("Failed:", failed)
    print("Total:", len(tests))

    print("====================================")
