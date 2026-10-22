"""Nonlinear least squares via :func:`scipy.optimize.least_squares`.

Three measures make this robust on badly scaled problems (see ADR 5):

1. The solver works on normalized parameters ``u = p / s`` with ``s = |p_start|``, so that all
   parameters are of order one.
2. The Jacobian uses central (3-point) finite differences. For Levenberg–Marquardt they are
   computed here, because older scipy versions silently fall back to 2-point for ``lm``.
3. After convergence, the solver restarts from the solution with fresh normalization until the
   cost no longer improves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from fitwise._typing import FloatArray
from fitwise.backends.base import BackendResult, FitProblem

_EPS_CBRT = float(np.finfo(float).eps ** (1 / 3))

_DESCRIPTIONS = {
    "trf": "Trust Region Reflective (scipy least_squares)",
    "lm": "Levenberg-Marquardt (MINPACK via scipy)",
}


@dataclass(frozen=True)
class LeastSquaresBackend:
    """Solve a :class:`FitProblem` with ``scipy.optimize.least_squares``.

    Parameters
    ----------
    method : {"trf", "lm"}
        Trust Region Reflective supports bounds; Levenberg–Marquardt does not.
    tol : float
        ``xtol``, ``ftol`` and ``gtol`` of each solver run.
    max_restarts : int
        Maximum number of restarts from the current solution.
    max_nfev : int or None
        Evaluation limit per solver run; ``None`` uses scipy's default.
    """

    method: Literal["trf", "lm"] = "trf"
    tol: float = 1e-15
    max_restarts: int = 10
    max_nfev: int | None = None

    def solve(self, problem: FitProblem) -> BackendResult:
        """Solve the problem, restarting until the cost stops improving."""
        names = problem.free_names
        if self.method == "lm" and problem.has_bounds:
            raise ValueError("Levenberg-Marquardt does not support bounds; use method='trf'")
        lo = np.array([problem.bounds[n][0] for n in names])
        hi = np.array([problem.bounds[n][1] for n in names])
        p = np.array([problem.init[n] for n in names])

        log: list[str] = []
        best_cost = np.inf
        result: OptimizeResult | None = None
        nfev = 0
        # The covariance uses the Jacobian of the *last* run: it started at the best point with
        # a normalization that matches that point. The first run's Jacobian is scaled by the
        # starting values, which can be off by many orders of magnitude.
        final_jac: FloatArray | None = None
        final_scale = _scale(p)
        for attempt in range(self.max_restarts + 1):
            scale = _scale(p)

            def residual(u: FloatArray, scale: FloatArray = scale) -> FloatArray:
                return problem.weighted_residual(dict(zip(names, u * scale, strict=True)))

            # Trial points may overflow; the solver rejects such steps by itself, and the final
            # result is checked separately, so floating-point warnings here are only noise.
            with np.errstate(all="ignore"):
                run = least_squares(
                    residual,
                    p / scale,
                    jac="3-point" if self.method == "trf" else partial(_central_jacobian, residual),
                    bounds=(lo / scale, hi / scale) if self.method == "trf" else (-np.inf, np.inf),
                    method=self.method,
                    xtol=self.tol,
                    ftol=self.tol,
                    gtol=self.tol,
                    max_nfev=self.max_nfev,
                )
            nfev += int(run.nfev)
            cost = 2 * float(run.cost)
            final_jac, final_scale = np.asarray(run.jac, dtype=float), scale
            if result is not None and not cost < best_cost * (1 - 1e-13):
                log.append(f"restart {attempt}: no further improvement (χ² = {cost:.10g})")
                break
            label = "initial run" if attempt == 0 else f"restart {attempt}"
            log.append(f"{label}: χ² = {cost:.10g}, {run.nfev} evals, status {run.status}")
            best_cost, result = cost, run
            p = np.clip(run.x * scale, lo, hi)

        assert result is not None
        assert final_jac is not None
        values = dict(zip(names, map(float, p), strict=True))
        cov_u = _covariance(final_jac)
        cov = None if cov_u is None else cov_u * np.outer(final_scale, final_scale)
        if cov is None:
            log.append("Jacobian is rank deficient; covariance not available")
        return BackendResult(
            values=values,
            cov_unscaled=cov,
            converged=bool(result.success),
            nfev=nfev,
            message=str(result.message),
            method=_DESCRIPTIONS[self.method],
            raw=result,
            log=tuple(log),
        )


def _central_jacobian(fun: Callable[[FloatArray], FloatArray], u: FloatArray) -> FloatArray:
    """Central finite differences with step ``eps^(1/3) * max(1, |u|)`` (no bounds)."""
    h = _EPS_CBRT * np.maximum(1.0, np.abs(u))
    columns = []
    for i, hi in enumerate(h):
        step = np.zeros_like(u)
        step[i] = hi
        columns.append((fun(u + step) - fun(u - step)) / (2 * hi))
    return np.column_stack(columns)


def _scale(p: FloatArray) -> FloatArray:
    return np.where(np.abs(p) > 0, np.abs(p), 1.0)


#: Relative singular value below which the Jacobian counts as rank deficient. Finite-difference
#: Jacobians of redundant parameters reach ~1e-12; with normalized parameters, the worst
#: legitimately ill-conditioned NIST problem (Nelson) stays at 3e-6.
RCOND = 1e-9


def _covariance(jac: FloatArray) -> FloatArray | None:
    """``(JᵀJ)⁻¹`` via SVD; ``None`` if ``J`` is numerically rank deficient."""
    if jac.size == 0 or not np.all(np.isfinite(jac)):
        return None
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    if s[-1] <= RCOND * s[0]:
        return None
    return np.asarray((vt.T / s**2) @ vt, dtype=float)
