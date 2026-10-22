"""Uncertainties beyond the covariance matrix: bootstrap and profile likelihood.

Both refit the model many times with the options of the original fit, starting from its
solution. They are slower than the covariance but do not assume that the model is linear
near the solution, so they reveal asymmetric uncertainties.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from scipy import optimize, stats

from fitwise._typing import FloatArray
from fitwise.report import format_measurement

if TYPE_CHECKING:
    from fitwise.result import FitResult

__all__ = ["BootstrapResult", "Interval", "ProfileResult", "bootstrap", "profile"]

#: Stand-in for χ² where a refit is impossible (model not finite).
_EXCLUDED = 1e300

#: Coverage of a one-sigma interval of a normal distribution.
ONE_SIGMA = float(stats.norm.cdf(1) - stats.norm.cdf(-1))


class Interval(NamedTuple):
    """A confidence interval ``[lower, upper]`` around ``value``; ±inf if open."""

    value: float
    lower: float
    upper: float
    level: float

    @property
    def minus(self) -> float:
        """Distance from the value down to the lower limit."""
        return self.value - self.lower

    @property
    def plus(self) -> float:
        """Distance from the value up to the upper limit."""
        return self.upper - self.value


def _refit(
    result: FitResult,
    *,
    x: FloatArray | None = None,
    y: FloatArray | None = None,
    yerr: FloatArray | None = None,
    xerr: FloatArray | None = None,
    fixed: dict[str, float] | None = None,
) -> FitResult:
    """Fit again with the original options, starting from the original solution."""
    from fitwise.api import fit

    data = result.data
    opts: dict[str, Any] = dict(result.options)
    fixed_all = {**opts.get("fixed", {}), **(fixed or {})}
    start = {n: result.params[n].value for n in result.free_names if n not in fixed_all}
    bounds = {
        n: tuple(None if math.isinf(b) else b for b in pair)
        for n, pair in opts.get("bounds", {}).items()
        if n not in fixed_all
    }
    method = opts.get("method", "auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(
            result.model,
            data.x if x is None else x,
            data.y if y is None else y,
            yerr=data.yerr if yerr is None else yerr,
            xerr=data.xerr if xerr is None else xerr,
            p0=start,
            bounds=bounds,  # type: ignore[arg-type]
            fixed=fixed_all,
            loss=opts.get("loss", "linear"),
            loss_scale=opts.get("loss_scale"),
            method="auto" if method == "multistart" else method,
            scale_covar=opts.get("scale_covar"),
        )


# --- bootstrap ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """Parameter distributions from refitting resampled data.

    Attributes
    ----------
    result : FitResult
        The original fit.
    samples : ndarray
        Shape ``(n_ok, k)``: refitted values of the free parameters, in ``result.free_names``
        order, for every resample that converged.
    method : {"parametric", "residuals", "cases"}
        How the data were resampled.
    n_failed : int
        Resamples whose refit did not converge; they are excluded.
    """

    result: FitResult
    samples: FloatArray
    method: str
    n_failed: int

    @property
    def std(self) -> dict[str, float]:
        """Standard deviation of each parameter over the resamples."""
        sd = self.samples.std(axis=0, ddof=1)
        return dict(zip(self.result.free_names, map(float, sd), strict=True))

    @property
    def covariance(self) -> FloatArray:
        """Covariance of the free parameters over the resamples."""
        return np.atleast_2d(np.cov(self.samples, rowvar=False))

    def interval(self, name: str, level: float = ONE_SIGMA) -> Interval:
        """Percentile interval of one parameter (default: central 68.3 %)."""
        i = self.result.free_names.index(name)
        lo, hi = np.percentile(self.samples[:, i], [50 * (1 - level), 50 * (1 + level)])
        return Interval(self.result.params[name].value, float(lo), float(hi), level)

    def __str__(self) -> str:
        n = len(self.samples)
        lines = [
            f"Bootstrap ({self.method}): {n} resamples"
            + (f", {self.n_failed} failed and excluded" if self.n_failed else ""),
            "",
            f"{'Parameter':<12}{'Value':>12}{'σ (cov.)':>12}{'σ (boot)':>12}   68.3 % interval",
        ]
        for name in self.result.free_names:
            p = self.result.params[name]
            value, err = format_measurement(p.value, p.stderr)
            _, boot = format_measurement(p.value, self.std[name])
            iv = self.interval(name)
            lines.append(
                f"{name:<12}{value:>12}{err:>12}{boot:>12}   [{iv.lower:.6g}, {iv.upper:.6g}]"
            )
        return "\n".join(lines)


def bootstrap(
    result: FitResult,
    n: int = 1000,
    *,
    method: Literal["auto", "parametric", "residuals", "cases"] = "auto",
    seed: int = 0,
) -> BootstrapResult:
    """Refit ``n`` resampled data sets.

    Parameters
    ----------
    result : FitResult
        The fit to examine.
    n : int
        Number of resamples.
    method : {"auto", "parametric", "residuals", "cases"}
        ``"parametric"`` adds Gaussian noise of size ``yerr`` to the fitted curve, consistent
        with absolute uncertainties. ``"residuals"`` adds resampled residuals instead, so the
        observed scatter counts. ``"cases"`` resamples data points with replacement, which also
        captures model misspecification but needs enough points. ``"auto"`` chooses
        ``"parametric"`` if ``yerr`` is taken as absolute and ``"residuals"`` otherwise, like
        the covariance (ADR 3).
    seed : int
        Seed of the random resampling.
    """
    if result.loss != "linear" or result.data.xerr is not None:
        raise NotImplementedError("bootstrap is available for ordinary least-squares fits")
    if n < 2:
        raise ValueError("n must be at least 2")
    if method == "auto":
        method = "parametric" if result.yerr is not None and not result.scale_covar else "residuals"
    if method not in ("parametric", "residuals", "cases"):
        raise ValueError(
            f"unknown method {method!r}; choose 'auto', 'parametric', 'residuals' or 'cases'"
        )
    if method == "parametric" and result.yerr is None:
        raise ValueError("the parametric bootstrap needs yerr")
    rng = np.random.default_rng(seed)
    data = result.data
    sigma = data.yerr if data.yerr is not None else np.ones(data.n)
    fitted = np.broadcast_to(result.eval(), (data.n,))
    normalized = (data.y - fitted) / sigma
    # residuals shrink by about sqrt(ndf / n); undo that so the resamples are not too narrow
    normalized = normalized * math.sqrt(data.n / result.ndf) if result.ndf > 0 else normalized

    samples, failed = [], 0
    for _ in range(n):
        if method == "parametric":
            refit = _refit(result, y=fitted + rng.normal(0.0, sigma))
        elif method == "residuals":
            y_new = fitted + rng.choice(normalized, data.n) * sigma
            refit = _refit(result, y=y_new)
        else:
            idx = rng.integers(0, data.n, data.n)
            refit = _refit(
                result,
                x=data.x[idx],
                y=data.y[idx],
                yerr=None if data.yerr is None else data.yerr[idx],
            )
        if refit.converged:
            samples.append([refit.params[k].value for k in result.free_names])
        else:
            failed += 1
    if len(samples) < 2:
        raise RuntimeError("fewer than two resamples converged")
    return BootstrapResult(result, np.array(samples), method, failed)


# --- profile likelihood ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ProfileResult:
    """Profile-likelihood intervals of the free parameters.

    Attributes
    ----------
    result : FitResult
        The original fit.
    intervals : dict
        ``{name: Interval}``.
    delta_chi2 : float
        The χ² increase that defines the interval edges.
    """

    result: FitResult
    intervals: dict[str, Interval]
    delta_chi2: float

    def __getitem__(self, name: str) -> Interval:
        return self.intervals[name]

    def __str__(self) -> str:
        level = next(iter(self.intervals.values())).level if self.intervals else ONE_SIGMA
        lines = [
            f"Profile likelihood: {100 * level:.1f} % intervals (Δχ² = {self.delta_chi2:.4g})",
            "",
            f"{'Parameter':<12}{'Value':>12}{'− err':>12}{'+ err':>12}{'σ (cov.)':>12}",
        ]
        for name, iv in self.intervals.items():
            p = self.result.params[name]
            ref = p.stderr if p.stderr else max(iv.minus, iv.plus)
            value, _ = format_measurement(p.value, ref)
            _, minus = (
                format_measurement(p.value, iv.minus) if math.isfinite(iv.minus) else (0, "∞")
            )
            _, plus = format_measurement(p.value, iv.plus) if math.isfinite(iv.plus) else (0, "∞")
            _, cov = format_measurement(p.value, p.stderr)
            lines.append(f"{name:<12}{value:>12}{minus:>12}{plus:>12}{cov:>12}")
        return "\n".join(lines)


def profile(
    result: FitResult,
    names: list[str] | None = None,
    *,
    level: float = ONE_SIGMA,
    max_steps: int = 30,
) -> ProfileResult:
    """Profile-likelihood confidence intervals.

    For each parameter, the interval contains all values for which χ², minimized over the
    other parameters, rises by at most Δχ² = χ²₁⁻¹(level) (1 for 68.3 %). Without ``yerr``,
    Δχ² is scaled by χ²/ndf. For a model that is linear in its parameters this reproduces
    value ± σ; for nonlinear models the interval can be asymmetric. Interval edges that are
    never reached within ``max_steps`` widening steps are reported as ±inf.
    """
    if result.loss != "linear":
        raise NotImplementedError("profile likelihood needs an ordinary χ², not a robust loss")
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    names = list(result.free_names if names is None else names)
    for n in names:
        if n not in result.free_names:
            raise ValueError(f"{n!r} is not a free parameter of this fit")
    delta = float(stats.chi2.ppf(level, 1))
    if result.scale_covar:
        if result.redchi is None:
            raise ValueError("no degrees of freedom left to scale Δχ²")
        delta *= result.redchi
    target = result.chi2 + delta

    intervals = {}
    for name in names:
        p = result.params[name]
        lo_bound, hi_bound = p.bounds
        step0 = p.stderr if p.stderr else (abs(p.value) * 0.1 or 1.0)

        def excess(v: float, name: str = name) -> float:
            # where the model cannot even be evaluated, the value is certainly excluded; a
            # huge finite number keeps the root bracketing of brentq intact
            try:
                refit = _refit(result, fixed={name: v})
            except ValueError:
                return _EXCLUDED
            return refit.chi2 - target if math.isfinite(refit.chi2) else _EXCLUDED

        edges = []
        for direction, bound in ((-1, lo_bound), (+1, hi_bound)):
            step, inner, edge = step0, p.value, math.inf * direction
            for _ in range(max_steps):
                outer = p.value + direction * step
                if (direction < 0 and outer <= bound) or (direction > 0 and outer >= bound):
                    outer = bound
                if excess(outer) > 0:
                    a, b = sorted((inner, outer))
                    edge = optimize.brentq(excess, a, b, xtol=1e-6 * step0, rtol=1e-10)
                    break
                if outer == bound:
                    edge = bound  # the interval extends to the bound
                    break
                inner, step = outer, step * 2
            edges.append(edge)
        intervals[name] = Interval(p.value, edges[0], edges[1], level)
    return ProfileResult(result, intervals, delta)
