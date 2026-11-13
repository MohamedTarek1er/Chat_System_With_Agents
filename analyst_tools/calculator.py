import statistics

def calculate(operation, numbers):

    if not numbers:
        raise ValueError("Numbers list cannot be empty.")

    if operation == "add":
        return sum(numbers)

    elif operation == "subtract":
        result = numbers[0]

        for number in numbers[1:]:
            result -= number

        return result

    elif operation == "multiply":
        result = 1

        for number in numbers:
            result *= number

        return result

    elif operation == "divide":
        result = numbers[0]

        for number in numbers[1:]:
            if number == 0:
                raise ValueError("Cannot divide by zero.")

            result /= number

        return result

    elif operation == "average":
        return statistics.mean(numbers)

    elif operation == "median":
        return statistics.median(numbers)

    elif operation == "percentage":
        if len(numbers) != 2:
            raise ValueError("Provide value and total.")

        value, total = numbers

        if total == 0:
            raise ValueError("Total cannot be zero.")

        return (value / total) * 100

    elif operation == "ratio":
        if len(numbers) != 2:
            raise ValueError("Provide exactly two numbers.")

        return f"{numbers[0]}:{numbers[1]}"

    else:
        raise ValueError("Unsupported operation.")
