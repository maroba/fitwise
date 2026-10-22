"""Models written as text, such as ``"a*exp(-x/tau) + c"``.

The expression is parsed with :mod:`ast` and checked against an allowlist: numbers, names,
the operators ``+ - * / **``, and a fixed set of mathematical functions. Anything else, such as
attribute access, indexing, comprehensions or unknown functions, is rejected before any code
runs. The checked expression is compiled once and evaluated with NumPy.

As in Origin, Excel and MATLAB, ``^`` means power: ``a*x^2`` is ``a*x**2``.
"""

from __future__ import annotations

import ast
import keyword
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import special

from fitwise._typing import ArrayLike, FloatArray
from fitwise.model import Model

__all__ = ["FUNCTIONS", "ExpressionModel", "parse_expression"]

#: Functions available in expressions.
FUNCTIONS: dict[str, Callable[..., Any]] = {
    **{
        name: getattr(np, name)
        for name in (
            "exp", "expm1", "exp2", "log", "log10", "log2", "log1p", "sqrt", "abs", "sign",
            "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh",
            "tanh", "arcsinh", "arccosh", "arctanh", "hypot", "heaviside", "minimum",
            "maximum", "floor", "ceil",
        )
    },
    "ln": np.log,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "erf": special.erf,
    "erfc": special.erfc,
}  # fmt: skip

#: Named constants available in expressions.
CONSTANTS: dict[str, float] = {"pi": float(np.pi)}

_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


def _reject(node: ast.AST, what: str, expr: str) -> ValueError:
    col = getattr(node, "col_offset", None)
    where = f" at position {col + 1}" if col is not None else ""
    return ValueError(f"{what}{where} in expression {expr!r}")


def parse_expression(expr: str, independent: str = "x") -> tuple[ast.Expression, list[str]]:
    """Validate ``expr`` and return its syntax tree and parameter names (in order of appearance).

    Raises
    ------
    ValueError
        If the expression uses anything outside the allowlist, has no parameters, or does not
        contain the independent variable.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("the expression must be a non-empty string")
    # "^" must become "**" before parsing: as an operator it binds weaker than "+" and "*",
    # so a + b*x^2 would parse as (a + b*x)^2. String literals are rejected below, so a plain
    # text replacement cannot change their meaning.
    source = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"invalid expression {expr!r}: {err.msg}") from None

    names: list[ast.Name] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, *_BINARY, *_UNARY)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINARY):
                raise _reject(node, f"operator {type(node.op).__name__} is not allowed", expr)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY):
                raise _reject(node, f"operator {type(node.op).__name__} is not allowed", expr)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise _reject(node, "only plain function calls like exp(x) are allowed", expr)
            if node.func.id not in FUNCTIONS:
                raise _reject(
                    node,
                    f"unknown function {node.func.id!r}; available: {', '.join(sorted(FUNCTIONS))}",
                    expr,
                )
        elif isinstance(node, ast.Name):
            names.append(node)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise _reject(node, f"constant {node.value!r} is not a number", expr)
        else:
            raise _reject(node, f"{type(node).__name__} is not allowed", expr)

    callees = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    variables = [n for n in names if id(n) not in callees]
    for node in variables:
        if node.id in FUNCTIONS:
            raise _reject(node, f"{node.id!r} is a function and must be called", expr)
    ordered = sorted(variables, key=lambda n: (n.lineno, n.col_offset))
    params: list[str] = []
    for node in ordered:
        if node.id not in (independent, *CONSTANTS, *params):
            params.append(node.id)
    if not any(node.id == independent for node in variables):
        raise ValueError(
            f"expression {expr!r} does not contain the independent variable {independent!r}"
        )
    if not params:
        raise ValueError(f"expression {expr!r} has no parameters to fit")
    for p in params:
        if keyword.iskeyword(p):
            raise ValueError(f"parameter name {p!r} is a Python keyword")
    return tree, params


class ExpressionModel(Model):
    """A :class:`~fitwise.Model` defined by a text expression.

    Parameters
    ----------
    expression : str
        For example ``"a*exp(-x/tau) + c"``. Parameters are all names other than the
        independent variable, the functions in :data:`FUNCTIONS` and ``pi``, in order of first
        appearance.
    independent : str
        Name of the independent variable.
    name : str, optional
        Name for the report; defaults to ``"expression"``.

    Examples
    --------
    >>> m = ExpressionModel("a*exp(-t/tau) + c", independent="t")
    >>> m.param_names
    ('a', 'tau', 'c')
    >>> float(m(0.0, a=2.0, tau=1.0, c=0.5))
    2.5
    """

    def __init__(self, expression: str, *, independent: str = "x", name: str | None = None):
        tree, params = parse_expression(expression, independent)
        code = compile(tree, filename="<fitwise expression>", mode="eval")
        namespace = {"__builtins__": {}, **FUNCTIONS, **CONSTANTS}

        def func(x: ArrayLike, *values: float) -> FloatArray:
            local = {
                independent: np.asarray(x, dtype=float),
                **dict(zip(params, values, strict=True)),
            }
            return np.asarray(
                eval(code, namespace, local), dtype=float
            )  # validated by parse_expression

        super().__init__(
            func, name=name or "expression", param_names=params, abbrev="x", formula=expression
        )
        self.expression = expression.strip()
        self.independent = independent

    def signature(self) -> str:
        """Return the expression as signature, e.g. ``y = a*exp(-x/tau) + c``."""
        return f"y = {self.expression}"
