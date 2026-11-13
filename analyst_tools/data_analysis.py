import pandas as pd
import matplotlib.pyplot as plt

def data_analysis(llm, question, evidence):

    context = "\n\n".join(
        str(item) for item in evidence
    )

    prompt = f"""
    You are a data analysis assistant.

    User Question:
    {question}

    Available Evidence:
    {context}

    Analyze the available information.

    Your tasks:
    1. Understand and organize the information.
    2. Compare data from different documents.
    3. Identify patterns and important findings.
    4. Identify similarities and differences.
    5. Perform statistical analysis when numerical data exists.
    6. Mention missing information and limitations.

    Do not invent information or numbers.
    Do not force numerical analysis on textual evidence.

    Explain your findings clearly.
    """

    response = llm.invoke(prompt)

    return response.content
