"""Direct solution of models that are linear in their parameters.

A model is linear (affine) in its free parameters if ``f(p) = f(0) + A p`` for all ``p``.
The design matrix ``A`` is obtained from ``k + 1`` model evaluations, the weighted least-squares
problem is solved with column-normalized SVD plus one step of iterative refinement, and no
starting values are needed. Accuracy on the NIST StRD linear suite: see ADR 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from fitwise._typing import FloatArray
from fitwise.backends.base import BackendResult, FitProblem

#: Relative singular value below which the design matrix counts as rank deficient. The matrix
#: is exact up to rounding here, so the threshold is far below that of the nonlinear backend.
RCOND = 1e-13

#: Relative tolerance of the affinity check ``f(p) == f(0) + A p``.
LINEARITY_RTOL = 1e-8


def design_matrix(problem: FitProblem) -> tuple[FloatArray, FloatArray]:
    """Return ``(A, f0)`` with ``f(p) = f0 + A p`` for the free parameters (if linear)."""
    names = problem.free_names
    n = problem.data.n

    def model(values: FloatArray) -> FloatArray:
        params = problem.all_params(dict(zip(names, values, strict=True)))
        with np.errstate(all="ignore"):
            out = problem.model.eval(problem.data.x, params)
        return np.broadcast_to(np.asarray(out, dtype=float), (n,))

    k = len(names)
    f0 = model(np.zeros(k))
    with np.errstate(all="ignore"):
        columns = [model(np.eye(k)[i]) - f0 for i in range(k)]
    return np.column_stack(columns) if columns else np.zeros((n, 0)), f0


def is_linear(problem: FitProblem, *, seed: int = 0) -> bool:
    """Check numerically whether the model is affine in its free parameters."""
    if problem.model.linear is not None:
        return problem.model.linear
    names = problem.free_names
    a, f0 = design_matrix(problem)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f0))):
        return False
    if np.any(np.all(a == 0, axis=0)):
        # A parameter without any effect cannot be confirmed as linear: a Gaussian far outside
        # the data range is exactly zero and would look like a constant model.
        return False
    rng = np.random.default_rng(seed)
    start = np.array([problem.init[n] for n in names])
    scale = np.maximum(np.abs(start), 1.0)
    for _ in range(2):
        p = start + scale * rng.standard_normal(len(names))
        params = problem.all_params(dict(zip(names, p, strict=True)))
        with np.errstate(all="ignore"):
            actual = np.broadcast_to(problem.model.eval(problem.data.x, params), f0.shape)
            predicted = f0 + a @ p
            size = np.max(np.abs(f0) + np.abs(a) @ np.abs(p))
            if not np.all(np.abs(actual - predicted) <= LINEARITY_RTOL * max(size, 1e-300)):
                return False
    return True


@dataclass(frozen=True)
class LinearBackend:
    """Weighted linear least squares, optionally with bounds."""

    def solve(self, problem: FitProblem) -> BackendResult:
        """Solve directly; no starting values or iterations are involved."""
        names = problem.free_names
        a, f0 = design_matrix(problem)
        w = problem.data.weights
        aw = a * w[:, None]
        bw = (problem.data.y - f0) * w
        norms = np.linalg.norm(aw, axis=0)
        norms[norms == 0] = 1.0
        an = aw / norms

        log = [f"design matrix from {len(names) + 1} model evaluations"]
        if problem.has_bounds:
            lo = np.array([problem.bounds[n][0] for n in names]) * norms
            hi = np.array([problem.bounds[n][1] for n in names]) * norms
            run = lsq_linear(an, bw, bounds=(lo, hi), method="bvls", tol=1e-14)
            u = np.asarray(run.x, dtype=float)
            method = "Bounded linear least squares (scipy lsq_linear)"
            converged, message, raw = bool(run.success), str(run.message), run
        else:
            u, *_ = np.linalg.lstsq(an, bw, rcond=None)
            u = u + np.linalg.lstsq(an, bw - an @ u, rcond=None)[0]  # iterative refinement
            log.append("one step of iterative refinement")
            method = "Linear least squares (direct)"
            converged, message, raw = True, "solved directly", None

        cov_u = _covariance(an)
        if cov_u is None:
            log.append("design matrix is rank deficient; covariance not available")
        return BackendResult(
            values=dict(zip(names, map(float, u / norms), strict=True)),
            cov_unscaled=None if cov_u is None else cov_u / np.outer(norms, norms),
            converged=converged,
            nfev=len(names) + 1,
            message=message,
            method=method,
            raw=raw,
            log=tuple(log),
        )


def _covariance(an: FloatArray) -> FloatArray | None:
    if an.size == 0 or not np.all(np.isfinite(an)):
        return None
    _, s, vt = np.linalg.svd(an, full_matrices=False)
    if s[-1] <= RCOND * s[0]:
        return None
    return np.asarray((vt.T / s**2) @ vt, dtype=float)
