"""Confidence and prediction bands of a fitted model (delta method)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from scipy import stats

from fitwise._typing import ArrayLike, FloatArray

if TYPE_CHECKING:
    from fitwise.result import FitResult

__all__ = ["Band", "confidence_band", "model_jacobian", "prediction_band"]

_EPS_CBRT = float(np.finfo(float).eps ** (1 / 3))


class Band(NamedTuple):
    """A band around the fitted curve.

    ``x`` is where the band is evaluated, ``center`` the fitted model there, ``lower`` and
    ``upper`` the band limits, ``level`` the coverage probability (e.g. 0.95), and ``kind``
    says what the band covers: ``"confidence"`` (the true curve) or ``"prediction"`` (a new
    measurement).
    """

    x: FloatArray
    center: FloatArray
    lower: FloatArray
    upper: FloatArray
    level: float
    kind: Literal["confidence", "prediction"]


def model_jacobian(result: FitResult, x: ArrayLike) -> FloatArray:
    """Differentiate the fitted model at ``x`` with respect to the free parameters.

    Central differences with a step relative to each parameter's magnitude, so that
    parameters of any scale (for example 1e-12) are differentiated accurately.
    """
    values = result.params.values_dict()
    xa = np.asarray(x, dtype=float)
    n = len(xa)
    columns = []
    for name in result.free_names:
        v = values[name]
        h = _EPS_CBRT * (abs(v) if v != 0 else 1.0)
        plus, minus = dict(values), dict(values)
        plus[name], minus[name] = v + h, v - h
        with np.errstate(all="ignore"):
            f_plus = np.broadcast_to(result.model.eval(xa, plus), (n,))
            f_minus = np.broadcast_to(result.model.eval(xa, minus), (n,))
        columns.append((f_plus - f_minus) / (2 * h))
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def _quantile(result: FitResult, level: float) -> float:
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    p = (1 + level) / 2
    if result.scale_covar and result.ndf > 0:
        return float(stats.t.ppf(p, result.ndf))  # scatter estimated from the residuals
    return float(stats.norm.ppf(p))  # uncertainties taken as known


def _curve_variance(result: FitResult, x: FloatArray) -> FloatArray:
    if result.covariance is None:
        raise ValueError("bands need the covariance matrix, which is not available for this fit")
    jac = model_jacobian(result, x)
    return np.asarray(np.einsum("ij,jk,ik->i", jac, result.covariance, jac), dtype=float)


def confidence_band(result: FitResult, x: ArrayLike, level: float = 0.95) -> Band:
    """Band that contains the *true curve* with probability ``level`` at each ``x``."""
    xa = np.asarray(x, dtype=float)
    center = np.broadcast_to(result.eval(xa), (len(xa),)).astype(float)
    half = _quantile(result, level) * np.sqrt(_curve_variance(result, xa))
    return Band(xa, center, center - half, center + half, level, "confidence")


def prediction_band(
    result: FitResult, x: ArrayLike, level: float = 0.95, *, yerr_new: ArrayLike | None = None
) -> Band:
    """Band that contains a *new measurement* with probability ``level`` at each ``x``.

    The measurement scatter is ``yerr_new`` if given; otherwise ``yerr`` of the fit if it is
    the same for all points, or the residual scatter if the fit had no ``yerr``. With
    ``scale_covar``, the scatter is scaled by χ²/ndf like the covariance.
    """
    xa = np.asarray(x, dtype=float)
    if yerr_new is not None:
        noise = np.broadcast_to(np.asarray(yerr_new, dtype=float), xa.shape) ** 2
    elif result.yerr is not None:
        if np.ptp(result.yerr) > 0:
            raise ValueError(
                "yerr differs between the data points; pass yerr_new= with the uncertainty "
                "of a new measurement at x"
            )
        noise = np.full(xa.shape, float(result.yerr[0]) ** 2)
    else:
        noise = np.ones(xa.shape)
    if result.scale_covar:
        if result.redchi is None:
            raise ValueError("no degrees of freedom left to estimate the scatter")
        noise = noise * result.redchi
    center = np.broadcast_to(result.eval(xa), (len(xa),)).astype(float)
    half = _quantile(result, level) * np.sqrt(_curve_variance(result, xa) + noise)
    return Band(xa, center, center - half, center + half, level, "prediction")


def describe(band: Band) -> str:
    """Short label such as ``95 % confidence band``."""
    return f"{100 * band.level:g} % {band.kind} band"
