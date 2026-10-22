"""Orthogonal distance regression for uncertainties in x and y, via ``odrpack`` (ADR 2).

Two stages: an ordinary fit in y (with all automatic choices for starting values) provides
the starting point, then ODRPACK95 minimizes Σ((y − f(x + δ))/yerr)² + Σ(δ/xerr)² over the
parameters and the corrections δ of x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fitwise._typing import FloatArray
from fitwise.backends.base import Backend, BackendResult, FitProblem


def _odrpack() -> Any:
    try:
        import odrpack
    except ImportError:
        raise ImportError(
            "fits with xerr need the odrpack package: pip install 'fitwise[odr]'"
        ) from None
    return odrpack


@dataclass(frozen=True)
class OdrBackend:
    """Ordinary fit with ``base``, then orthogonal distance regression from its solution."""

    base: Backend
    maxit: int = 200

    def solve(self, problem: FitProblem) -> BackendResult:
        """Solve in two stages and return the ODR result."""
        odrpack = _odrpack()
        data = problem.data
        if data.xerr is None or data.yerr is None:
            raise ValueError("orthogonal distance regression needs both xerr and yerr")
        first = self.base.solve(problem)
        names = problem.free_names

        def f(x: FloatArray, beta: FloatArray) -> FloatArray:
            params = problem.all_params(dict(zip(names, beta, strict=True)))
            out = problem.model.eval(x, params)
            return np.asarray(np.broadcast_to(out, x.shape), dtype=float)

        lo = np.array([problem.bounds[n][0] for n in names])
        hi = np.array([problem.bounds[n][1] for n in names])
        beta0 = np.clip([first.values[n] for n in names], lo, hi)
        with np.errstate(all="ignore"):
            run = odrpack.odr_fit(
                f,
                data.x,
                data.y,
                beta0,
                weight_x=1.0 / data.xerr**2,
                weight_y=1.0 / data.yerr**2,
                bounds=(lo, hi) if problem.has_bounds else None,
                diff_scheme="central",
                maxit=self.maxit,
            )
        # ODRPACK reports the covariance without the factor res_var (= χ²/ndf), as scipy.odr
        cov = np.asarray(run.cov_beta, dtype=float)
        # irank is ODRPACK's rank *deficiency* of the Jacobian: 0 means full rank
        full_rank = int(run.irank) == 0 and bool(np.all(np.isfinite(cov)))
        cov_unscaled = cov if full_rank else None
        # ODRPACK's info: last digit 1-3 = convergence criterion met; leading digit 4 =
        # questionable (e.g. rank deficient, reported separately), 5 or more = fatal error
        info = int(run.info)
        converged = info % 10 in (1, 2, 3) and info // 10000 < 5
        message = str(run.stopreason)
        on_bound = [
            n
            for n, b, low, high in zip(names, run.beta, lo, hi, strict=True)
            if b <= low or b >= high
        ]
        if not converged and on_bound:
            message += (
                f" The solution lies on a bound ({', '.join(on_bound)}), which ODRPACK cannot "
                "handle. Widen the bound, or hold the parameter at the bound value with fixed=."
            )
        log = (
            f"stage 1 (ordinary fit in y): {first.method}",
            *first.log,
            f"stage 2: orthogonal distance regression, {run.niter} iterations, "
            f"stop reason: {run.stopreason}",
        )
        return BackendResult(
            values=dict(zip(names, map(float, run.beta), strict=True)),
            cov_unscaled=cov_unscaled,
            converged=converged,
            nfev=first.nfev + int(run.nfev),
            message=message,
            method="Orthogonal distance regression (odrpack)",
            raw=run,
            log=log,
            chi2=float(run.sum_square),
        )
