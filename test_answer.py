
from langchain_core.documents import Document

from answer_tools.answer_agent import answer_question


# Original user question
question = "Which model achieved the highest accuracy?"


# Example Analyst results
analysis = """
Model A achieved 90% accuracy.
Model B achieved 95% accuracy.
Model C achieved 92% accuracy.

Model B achieved the highest accuracy with 95%.
"""


# Example evidence
evidence = [

    Document(
        page_content="Model A achieved 90% accuracy.",
        metadata={
            "document_name": "research_1.pdf",
            "page_number": 3
        }
    ),

    Document(
        page_content="Model B achieved 95% accuracy.",
        metadata={
            "document_name": "research_2.pdf",
            "page_number": 5
        }
    ),

    Document(
        page_content="Model C achieved 92% accuracy.",
        metadata={
            "document_name": "research_3.pdf",
            "page_number": 7
        }
    )
]


# Run Answer Agent
result = answer_question(
    question,
    analysis,
    evidence
)


print("\nFINAL ANSWER")
print("=" * 50)
print(result)
