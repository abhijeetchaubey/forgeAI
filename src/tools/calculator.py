"""
Independent Tool 2: Safe Arithmetic Calculator Tool
Evaluates mathematical expressions using a strict AST whitelist.
Does NOT use eval(), exec(), or expose arbitrary Python execution.
"""

import ast
import math
import operator
from typing import Optional
from pydantic import BaseModel, Field

from src.tools.base import RegisteredTool


class CalculatorInput(BaseModel):
    """Input schema for safe calculator tool."""
    expression: str = Field(
        ...,
        min_length=1,
        max_length=250,
        description="Arithmetic expression to evaluate, e.g. '(120 * 1.15) + 50' or 'sqrt(144)'"
    )


class CalculatorOutput(BaseModel):
    """Output schema for safe calculator tool."""
    success: bool
    expression: str = ""
    result: Optional[float] = None
    error: Optional[str] = None


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _eval_ast_node(node: ast.AST) -> float:
    """Recursively evaluates only whitelisted arithmetic AST nodes."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        if op_type is ast.Pow and abs(right) > 1000:
            raise ValueError("Exponent out of safe bounds.")
        return float(_BINARY_OPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return float(_UNARY_OPS[op_type](_eval_ast_node(node.operand)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            func_name = getattr(node.func, "id", type(node.func).__name__)
            raise ValueError(f"Function '{func_name}' is not allowed.")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in calculator.")
        args = [_eval_ast_node(arg) for arg in node.args]
        return float(_SAFE_FUNCS[node.func.id](*args))

    raise ValueError(f"Disallowed expression construct: {type(node).__name__}")


def execute_calculator(params: CalculatorInput) -> CalculatorOutput:
    """Safely parses and evaluates an arithmetic expression via AST whitelisting."""
    expr = params.expression.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast_node(tree)
        return CalculatorOutput(
            success=True,
            expression=expr,
            result=round(value, 8),
        )
    except Exception as exc:
        return CalculatorOutput(
            success=False,
            expression=expr,
            error=f"Calculation error: {exc}",
        )


calculator_tool = RegisteredTool(
    name="calculator",
    description="Safely evaluates arithmetic expressions, unit conversions, percentages, totals, and cost calculations.",
    input_schema=CalculatorInput,
    output_schema=CalculatorOutput,
    execute=execute_calculator,
    use_cases=["calculate", "math", "arithmetic", "procurement", "budget", "cost", "total", "price", "quantity", "engineering", "conversion"],
)
