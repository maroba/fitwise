"""Robust fits that downweight outliers.

Two stages: an ordinary fit with any backend (direct, multi-start or TRF) provides the starting
point, then Trust Region Reflective minimizes a robust loss from there. The scale at which
points count as outliers is ``loss_scale`` in units of ``yerr`` (default 1.5) or, without
``yerr``, 1.5 times a robust estimate (MAD) of the residual scatter of the first stage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fitwise.backends.base import Backend, BackendResult, FitProblem
from fitwise.backends.least_squares import LeastSquaresBackend, Loss

#: Robust losses of :func:`scipy.optimize.least_squares`.
ROBUST_LOSSES = ("soft_l1", "huber", "cauchy", "arctan")

#: Default outlier scale in units of the (weighted) residual scatter.
DEFAULT_SCALE = 1.5

#: Points whose robust weight is below this count as downweighted in the report.
DOWNWEIGHTED = 0.5


@dataclass(frozen=True)
class RobustBackend:
    """Ordinary fit with ``base``, then a robust TRF fit starting from its solution."""

    base: Backend
    loss: Loss
    loss_scale: float | None = None

    def solve(self, problem: FitProblem) -> BackendResult:
        """Solve in two stages and return the robust result."""
        first = self.base.solve(problem)
        with np.errstate(all="ignore"):
            resid = problem.weighted_residual(first.values)
        if self.loss_scale is not None:
            scale, origin = self.loss_scale, "given"
        elif problem.data.yerr is not None:
            scale, origin = DEFAULT_SCALE, "default, in units of yerr"
        else:
            mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
            scale = DEFAULT_SCALE * mad if mad > 0 else 1.0
            origin = "1.5 × MAD of the residuals of an ordinary fit"
        start = replace(problem, init=dict(first.values))
        second = LeastSquaresBackend("trf", loss=self.loss, f_scale=scale).solve(start)
        weights = second.robust_weights
        n_down = int(np.sum(weights < DOWNWEIGHTED)) if weights is not None else 0
        log = (
            f"stage 1 (ordinary fit): {first.method}",
            *first.log,
            f"stage 2: {self.loss} loss, scale {scale:.4g} ({origin}); "
            f"{n_down} of {problem.data.n} points downweighted below {DOWNWEIGHTED}",
            *second.log,
        )
        return replace(
            second,
            nfev=first.nfev + second.nfev,
            method=f"Robust fit ({self.loss} loss, scale {scale:.3g}) with Trust Region Reflective",
            log=log,
        )
