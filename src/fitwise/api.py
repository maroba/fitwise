"""The public entry point :func:`fit`."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from fitwise._typing import ArrayLike, Bounds
from fitwise._validate import (
    FitData,
    NanPolicy,
    as_float,
    check_names,
    normalize_bounds,
    validate_data,
)
from fitwise.backends import Backend, FitProblem, LeastSquaresBackend, LinearBackend
from fitwise.backends.linear import is_linear
from fitwise.model import Model, as_model
from fitwise.result import FitResult, build_result

__all__ = ["fit"]

Method = Literal["auto", "trf", "lm", "linear", "odr", "multistart"]

_PLANNED = {
    "multistart": "0.2",
    "odr": "0.3",
}


def fit(
    model: Model | Callable[..., ArrayLike] | str,
    x: ArrayLike,
    y: ArrayLike,
    *,
    yerr: ArrayLike | None = None,
    xerr: ArrayLike | None = None,
    p0: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    fixed: Mapping[str, float] | None = None,
    loss: str = "linear",
    method: Method = "auto",
    scale_covar: bool | None = None,
    nan_policy: NanPolicy = "raise",
) -> FitResult:
    """Fit a model to data.

    Parameters
    ----------
    model : Model or callable
        A built-in model from :mod:`fitwise.models`, a composition such as
        ``gaussian + linear``, or a function ``f(x, p1, p2, ...)``.
    x : array_like
        Independent variable, shape ``(n,)`` or ``(n, k)``; passed to the model unchanged.
    y : array_like
        Measured values, shape ``(n,)``.
    yerr : float or array_like, optional
        Absolute one-sigma uncertainties of ``y``. Without ``yerr``, uncertainties are
        estimated from the scatter of the residuals.
    xerr : array_like, optional
        Uncertainties of ``x`` (orthogonal distance regression; planned for 0.3).
    p0 : dict, optional
        Starting values ``{name: value}``; override any heuristic per parameter.
    bounds : dict, optional
        ``{name: (lo, hi)}``; use ``None`` for an open side.
    fixed : dict, optional
        ``{name: value}`` of parameters held constant.
    loss : str
        Only ``"linear"`` (ordinary least squares) is supported so far.
    method : {"auto", "linear", "trf", "lm"}
        Solver. ``"auto"`` solves models that are linear in their parameters directly (no
        starting values needed) and uses Trust Region Reflective otherwise. ``"lm"`` is
        Levenberg–Marquardt without bounds.
    scale_covar : bool, optional
        Multiply the covariance by χ²/ndf. Default: ``False`` with ``yerr`` (uncertainties are
        taken as absolute), ``True`` without.
    nan_policy : {"raise", "omit"}
        What to do with non-finite data points.

    Returns
    -------
    FitResult
        Parameters with uncertainties, goodness of fit, warnings and a printable report.

    Examples
    --------
    >>> import fitwise as fw
    >>> x = np.linspace(-5, 5, 101)
    >>> y = fw.models.gaussian(x, amp=2.0, cen=0.3, sigma=1.1)
    >>> result = fw.fit(fw.models.gaussian, x, y + 0.01 * np.cos(7 * x))
    >>> round(result.params["cen"].value, 2)
    0.3
    """
    m = as_model(model)
    if xerr is not None:
        raise NotImplementedError(
            "xerr (orthogonal distance regression) is planned for fitwise 0.3"
        )
    if loss != "linear":
        raise NotImplementedError("robust loss functions are planned for fitwise 0.2")
    data = validate_data(x, y, yerr, nan_policy)

    names = m.param_names
    p0_given = dict(p0 or {})
    fixed_given = dict(fixed or {})
    bounds_given = normalize_bounds(bounds)
    for argument, mapping in (("p0", p0_given), ("bounds", bounds_given), ("fixed", fixed_given)):
        check_names(argument, mapping, names)
    fixed_values = {n: as_float("fixed", n, v) for n, v in fixed_given.items()}
    for n, v in fixed_values.items():
        if not math.isfinite(v):
            raise ValueError(f"fixed[{n!r}] must be finite, got {v}")
        for argument, mapping in (("p0", p0_given), ("bounds", bounds_given)):
            if n in mapping:
                raise ValueError(f"parameter {n!r} is fixed and must not appear in {argument}")

    free = [n for n in names if n not in fixed_values]
    if not free:
        raise ValueError("all parameters are fixed; there is nothing to fit")
    if data.n < len(free):
        raise ValueError(f"{data.n} data points cannot determine {len(free)} free parameters")

    free_bounds = {n: bounds_given.get(n, (-math.inf, math.inf)) for n in free}
    init, strategies = _initial_values(m, data, free, p0_given, free_bounds)
    strategies.update(dict.fromkeys(fixed_values, "fixed"))
    problem = FitProblem(m, data, init, fixed_values, free_bounds)
    _check_start(problem, strategies)

    backend, reason = _select_backend(method, problem)
    if isinstance(backend, LinearBackend):
        strategies.update(dict.fromkeys(problem.free_names, "not needed (linear model)"))
    solved = backend.solve(problem)
    result = build_result(
        problem,
        solved,
        strategies=strategies,
        scale_covar=scale_covar,
        method_reason=reason,
        options={
            "p0": p0_given,
            "bounds": {n: list(b) for n, b in bounds_given.items()},
            "fixed": fixed_values,
            "method": method,
            "scale_covar": scale_covar,
            "nan_policy": nan_policy,
            "loss": loss,
        },
    )
    for w in result.warnings:
        warnings.warn(w, stacklevel=2)
    return result


def _initial_values(
    model: Model,
    data: FitData,
    free: list[str],
    p0: Mapping[str, Any],
    bounds: Mapping[str, Bounds],
) -> tuple[dict[str, float], dict[str, str]]:
    """Choose starting values by precedence: p0 > signature default > heuristic > fallback."""
    defaults = model.defaults
    needs_guess = any(n not in p0 and n not in defaults for n in free)
    guessed = model._guess_labeled(data.x, data.y) if needs_guess else {}

    init: dict[str, float] = {}
    strategies: dict[str, str] = {}
    for n in free:
        lo, hi = bounds[n]
        if n in p0:
            value, strategy = as_float("p0", n, p0[n]), "p0"
            if not lo <= value <= hi:
                raise ValueError(f"p0[{n!r}] = {value:g} is outside its bounds ({lo:g}, {hi:g})")
        elif n in defaults:
            value, strategy = defaults[n], "signature default"
        elif n in guessed:
            value, strategy = guessed[n]
        else:
            value, strategy = _fallback(lo, hi), "fallback"
        if not lo <= value <= hi:
            value = float(np.clip(value, lo, hi))
            strategy += " (clipped to bounds)"
        init[n], strategies[n] = value, strategy
    return init, strategies


def _fallback(lo: float, hi: float) -> float:
    if math.isfinite(lo) and math.isfinite(hi):
        return (lo + hi) / 2
    if math.isfinite(lo):
        return lo + max(1.0, abs(lo))
    if math.isfinite(hi):
        return hi - max(1.0, abs(hi))
    return 1.0


def _check_start(problem: FitProblem, strategies: Mapping[str, str]) -> None:
    """Fail early with a clear message if the model cannot be evaluated at the start."""
    n = problem.data.n
    start = problem.all_params(problem.init)
    with np.errstate(all="ignore"):
        model_y = problem.model.eval(problem.data.x, start)
    try:
        model_y = np.broadcast_to(model_y, (n,))
    except ValueError:
        raise ValueError(
            f"the model returned an array of shape {np.shape(model_y)}, expected ({n},)"
        ) from None
    if not np.all(np.isfinite(model_y)):
        values = ", ".join(f"{k} = {v:g} ({strategies[k]})" for k, v in start.items())
        raise ValueError(
            f"the model is not finite at the starting values {values}. "
            "Pass better starting values with p0= or restrict parameters with bounds=."
        )


def _select_backend(method: str, problem: FitProblem) -> tuple[Backend, str]:
    if method == "auto":
        if is_linear(problem):
            return LinearBackend(), "auto: linear in the parameters"
        return LeastSquaresBackend("trf"), "auto: nonlinear"
    if method == "linear":
        if not is_linear(problem):
            raise ValueError("method='linear' needs a model that is linear in its parameters")
        return LinearBackend(), "requested"
    if method == "trf":
        return LeastSquaresBackend("trf"), "requested"
    if method == "lm":
        if problem.has_bounds:
            raise ValueError("method='lm' does not support bounds; use 'trf' or 'auto'")
        return LeastSquaresBackend("lm"), "requested"
    if method in _PLANNED:
        raise NotImplementedError(f"method={method!r} is planned for fitwise {_PLANNED[method]}")
    raise ValueError(f"unknown method {method!r}; choose 'auto', 'linear', 'trf' or 'lm'")
