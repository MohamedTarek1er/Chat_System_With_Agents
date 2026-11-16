import pandas as pd
import matplotlib.pyplot as plt

from core.prompts import DATA_ANALYSIS_PROMPT


def data_analysis(llm, question, evidence):
    context = "\n\n".join(
        str(item) for item in evidence
    )

    prompt = DATA_ANALYSIS_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(prompt)

    return response.content
