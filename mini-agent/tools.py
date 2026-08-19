import ast
import operator


# Allowed mathematical operations.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    """Safely evaluate a mathematical expression AST."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        operation = _OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("This operator is not supported.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        # Prevent extremely large powers.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = _OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("This unary operator is not supported.")

        return operation(_evaluate(node.operand))

    raise ValueError("Invalid mathematical expression.")


def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to calculate.

    Returns:
        The calculated result as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return str(result)

    except ZeroDivisionError:
        return "Error: division by zero."

    except Exception as error:
        return f"Could not calculate that expression: {error}"


# Tool schema used by Ollama.
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Calculate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression such as "
                        "125 * 48 or (20 + 5) / 5."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}


TOOLS = [calculator_tool]

AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
}
