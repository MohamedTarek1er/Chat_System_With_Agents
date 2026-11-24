from agents.report.report_agent import generate_report


path = generate_report(
    question="Which model performed best?",
    answer="""
XGBoost achieved the highest accuracy with 95%.

Random Forest achieved 92%, while Logistic Regression
achieved 87%.
""",
    analysis="""
XGBoost performed better in terms of accuracy.
However, additional metrics should be considered before
choosing the final model.
""",
    sources=[
        {
            "doc_name": "Model_Evaluation.pdf",
            "page": 5
        },
        {
            "doc_name": "Results.pdf",
            "page": 8
        }
    ],
    style_request="Formal report, Times font, dark green headings, 12 point text."
)

print("Report generated:", path)