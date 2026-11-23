import statistics
from typing import Literal
from langchain_core.tools import tool

@tool
def calculate_tool(
    operation: Literal["add", "subtract", "multiply", "divide",
                    "average", "median", "percentage", "ratio"],
    numbers: list[float],
) -> str:
    """Exact arithmetic over a list of numbers. Use for ALL averages, differences,
    percentages and ratios. For 'percentage' pass [value, total]; for 'ratio' pass two numbers."""
    if not numbers:
        return "Error: numbers list cannot be empty."

    if operation == "add":
        return str(sum(numbers))
    if operation == "subtract":
        result = numbers[0]
        for n in numbers[1:]:
            result -= n
        return str(result)
    if operation == "multiply":
        result = 1
        for n in numbers:
            result *= n
        return str(result)
    if operation == "divide":
        result = numbers[0]
        for n in numbers[1:]:
            if n == 0:
                return "Error: cannot divide by zero."
            result /= n
        return str(result)
    if operation == "average":
        return str(statistics.mean(numbers))
    if operation == "median":
        return str(statistics.median(numbers))
    if operation in ("percentage", "ratio"):
        if len(numbers) != 2:
            return f"Error: '{operation}' needs exactly two numbers."
        a, b = numbers
        if operation == "percentage":
            return "Error: total cannot be zero." if b == 0 else str((a / b) * 100)
        return f"{a}:{b}"
    return "Error: unsupported operation."