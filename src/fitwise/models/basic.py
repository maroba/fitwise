"""Constant, linear and polynomial models (linear in their parameters)."""

from __future__ import annotations

import numpy as np

from fitwise._typing import ArrayLike, FloatArray
from fitwise.model import Model
from fitwise.models._heuristics import sorted_xy

__all__ = ["constant", "linear", "polynomial"]


def _constant(x: ArrayLike, c: float) -> FloatArray:
    return c + np.zeros_like(np.asarray(x, dtype=float))


def _linear(x: ArrayLike, slope: float, intercept: float) -> FloatArray:
    return slope * np.asarray(x, dtype=float) + intercept


def _guess_constant(x: FloatArray, y: FloatArray) -> dict[str, float]:
    return {"c": float(np.mean(y))}


def _guess_linear(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = sorted_xy(x, y)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return {"slope": 0.0, "intercept": float(np.mean(ys))}
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


constant = Model(
    _constant,
    name="constant",
    guess=_guess_constant,
    guess_label="mean",
    linear=True,
    shape="baseline",
    abbrev="c",
    formula=r"c",
    param_docs={"c": "constant offset"},
    dims={"c": (0, 1)},
)
"""Constant offset :math:`c`."""

linear = Model(
    _linear,
    name="linear",
    guess=_guess_linear,
    guess_label="linear least squares",
    linear=True,
    shape="baseline",
    abbrev="lin",
    formula=r"\mathrm{slope}\,x + \mathrm{intercept}",
    param_docs={"slope": "slope", "intercept": "value at x = 0"},
    dims={"slope": (-1, 1), "intercept": (0, 1)},
)
"""Straight line :math:`\\mathrm{slope}\\,x + \\mathrm{intercept}`."""


def polynomial(degree: int) -> Model:
    r"""Polynomial :math:`c_0 + c_1 x + \dots + c_n x^n` of the given degree.

    Parameters
    ----------
    degree : int
        Polynomial degree ``n >= 0``; the parameters are ``c0`` to ``cn``.

    Examples
    --------
    >>> polynomial(2).param_names
    ('c0', 'c1', 'c2')
    """
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError(f"degree must be a non-negative integer, got {degree!r}")
    names = [f"c{i}" for i in range(degree + 1)]

    def func(x: ArrayLike, *coeffs: float) -> FloatArray:
        xa = np.asarray(x, dtype=float)
        values: FloatArray = np.asarray(np.polynomial.polynomial.polyval(xa, coeffs), dtype=float)
        return np.asarray(values + np.zeros_like(xa), dtype=float)

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = sorted_xy(x, y)
        deg = min(degree, len(np.unique(xs)) - 1)
        coeffs = np.zeros(degree + 1)
        if deg >= 0:
            coeffs[: deg + 1] = np.polynomial.polynomial.polyfit(xs, ys, deg)
        return dict(zip(names, map(float, coeffs), strict=True))

    return Model(
        func,
        name=f"polynomial{degree}",
        param_names=names,
        guess=guess,
        guess_label="polynomial least squares",
        linear=True,
        shape="baseline",
        abbrev="p",
        formula=" + ".join(f"c_{i} x^{i}" for i in range(degree + 1)),
        param_docs={n: f"coefficient of x^{i}" for i, n in enumerate(names)},
        dims={n: (-i, 1) for i, n in enumerate(names)},
    )
