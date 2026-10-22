"""The result of a fit: parameters, statistics, warnings, and everything needed to repeat it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from fitwise import warnings as fw_warnings
from fitwise._typing import ArrayLike, FloatArray
from fitwise._validate import FitData
from fitwise.backends.base import BackendResult, FitProblem
from fitwise.model import Model
from fitwise.parameters import Parameter, Parameters

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from fitwise.bands import Band

__all__ = ["FitResult"]

#: Correlation above which a StrongCorrelationWarning is raised. Correlations of 0.95-0.99 are
#: normal for many physical laws (Arrhenius, power laws, lines far from x = 0) and are listed in
#: the report instead; above 0.999 only a combination of the parameters is determined.
STRONG_CORRELATION = 0.999
#: Two-sided probability outside of which χ² is considered implausible.
CHI2_ALPHA = 0.01


@dataclass(frozen=True, eq=False)
class FitResult:
    """Outcome of :func:`fitwise.fit`. Identical in structure for every solver.

    The most important attributes are :attr:`params`, :attr:`ok` and :attr:`warnings`;
    ``print(result)`` shows the full report.
    """

    model: Model
    """The fitted model."""
    params: Parameters
    """Fitted parameters with uncertainties, including fixed ones."""
    data: FitData
    """The data used in the fit (after ``nan_policy``)."""
    method: str
    """Solver that ran."""
    method_reason: str
    """Why this solver was chosen."""
    converged: bool
    """Whether the solver met its convergence criteria."""
    nfev: int
    """Number of model evaluations."""
    message: str
    """The solver's status message."""
    free_names: tuple[str, ...]
    """Names of the varied parameters; the order of :attr:`covariance`."""
    covariance: FloatArray | None
    """Covariance of the free parameters (scaled per :attr:`scale_covar`), or ``None``."""
    scale_covar: bool
    """Whether the covariance was multiplied by χ²/ndf."""
    chi2: float
    """Weighted sum of squared residuals Σ((y − f)/yerr)²; equals :attr:`rss` without yerr."""
    ndf: int
    """Degrees of freedom: points minus free parameters."""
    redchi: float | None
    """χ²/ndf, or ``None`` if ``ndf == 0``."""
    pvalue: float | None
    """Probability of a χ² at least this large if model and ``yerr`` are right (needs yerr)."""
    rss: float
    """Unweighted residual sum of squares."""
    residual_sd: float | None
    """Residual standard deviation √(rss/ndf)."""
    r2: float
    """Coefficient of determination, weighted with 1/yerr² if ``yerr`` is given."""
    aic: float
    """Akaike information criterion n·ln(χ²/n) + 2k."""
    bic: float
    """Bayesian information criterion n·ln(χ²/n) + k·ln(n)."""
    warnings: tuple[fw_warnings.FitWarning, ...]
    """Everything that makes the result questionable."""
    solver_log: tuple[str, ...]
    """What the solver did, step by step."""
    raw: object = field(repr=False)
    """The backend's native result object."""
    options: Mapping[str, Any] = field(default_factory=dict, repr=False)
    """The fit options as given by the user."""
    loss: str = "linear"
    """``"linear"`` for ordinary least squares, otherwise the robust loss used."""
    robust_weights: FloatArray | None = field(default=None, repr=False)
    """Robust weights at the solution (1 = normal point, ≪ 1 = outlier); ``None`` if not robust."""

    @property
    def downweighted(self) -> FloatArray:
        """Boolean mask of points a robust fit treated as outliers (weight below 0.5)."""
        if self.robust_weights is None:
            return np.zeros(self.data.n, dtype=bool)
        return np.asarray(self.robust_weights < 0.5)

    # --- convenience accessors ----------------------------------------------------------

    @property
    def ok(self) -> bool:
        """``True`` if the solver converged and parameter uncertainties are available."""
        return self.converged and (self.covariance is not None or not self.free_names)

    @property
    def x(self) -> FloatArray:
        """Independent variable."""
        return self.data.x

    @property
    def y(self) -> FloatArray:
        """Dependent variable."""
        return self.data.y

    @property
    def yerr(self) -> FloatArray | None:
        """Uncertainties of ``y``, if given."""
        return self.data.yerr

    @property
    def initial_guess(self) -> dict[str, float]:
        """Starting value of every parameter."""
        return {n: p.init_value for n, p in self.params.items()}

    @property
    def guess_strategy(self) -> dict[str, str]:
        """How each starting value was obtained."""
        return {n: p.guess_strategy for n, p in self.params.items()}

    @property
    def correlation(self) -> FloatArray | None:
        """Correlation matrix of the free parameters."""
        return None if self.covariance is None else _correlation(self.covariance)

    def correlations(self, threshold: float = 0.0) -> dict[tuple[str, str], float]:
        """Parameter pairs whose absolute correlation exceeds ``threshold``, strongest first."""
        corr = self.correlation
        if corr is None:
            return {}
        pairs = {
            (a, b): float(corr[i, j])
            for i, a in enumerate(self.free_names)
            for j, b in enumerate(self.free_names)
            if i < j and abs(corr[i, j]) > threshold
        }
        return dict(sorted(pairs.items(), key=lambda kv: -abs(kv[1])))

    def eval(self, x: ArrayLike | None = None) -> FloatArray:
        """Evaluate the fitted model at ``x`` (default: the data's ``x``)."""
        return self.model.eval(self.x if x is None else x, self.params.values_dict())

    @property
    def residuals(self) -> FloatArray:
        """``y − f(x)``."""
        return self.y - np.broadcast_to(self.eval(), self.y.shape)

    @property
    def normalized_residuals(self) -> FloatArray:
        """``(y − f(x)) / σ``; equals :attr:`residuals` without ``yerr``.

        σ is ``yerr``, or with ``xerr`` the effective uncertainty √(yerr² + (f′(x)·xerr)²).
        """
        sigma = self.effective_yerr
        return self.residuals if sigma is None else self.residuals / sigma

    @property
    def effective_yerr(self) -> FloatArray | None:
        """``yerr``, or √(yerr² + (f′(x)·xerr)²) if ``x`` has uncertainties too."""
        xerr = self.data.xerr
        if self.yerr is None or xerr is None:
            return self.yerr
        x = self.x
        h = 1e-6 * np.maximum(np.abs(x), 1.0) + 1e-6 * np.ptp(x)
        slope = (self.eval(x + h) - self.eval(x - h)) / (2 * h)
        return np.asarray(np.hypot(self.yerr, slope * xerr), dtype=float)

    # --- output -------------------------------------------------------------------------

    def __str__(self) -> str:
        from fitwise.report import text_report

        return text_report(self)

    def __repr__(self) -> str:
        status = "ok" if self.ok else "NOT OK"
        quality = (
            f"χ²/ndf={self.redchi:.3g}" if self.redchi is not None and self.yerr is not None else ""
        )
        n_warn = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return f"<FitResult {self.model.name}: {status}{', ' + quality if quality else ''}{n_warn}>"

    def confidence_band(self, x: ArrayLike, level: float = 0.95) -> Band:
        """Band containing the true curve with probability ``level``; see :mod:`fitwise.bands`."""
        from fitwise.bands import confidence_band

        return confidence_band(self, x, level)

    def prediction_band(
        self, x: ArrayLike, level: float = 0.95, *, yerr_new: ArrayLike | None = None
    ) -> Band:
        """Band containing a new measurement with probability ``level``."""
        from fitwise.bands import prediction_band

        return prediction_band(self, x, level, yerr_new=yerr_new)

    def plot(
        self,
        *,
        residuals: bool = True,
        band: Literal["confidence", "prediction"] | None = "confidence",
        level: float = 0.95,
        ax: Axes | None = None,
        show_params: bool = False,
    ) -> tuple[Figure, list[Axes]]:
        """Plot data, fit, band and residuals; see :func:`fitwise.plotting.plot_result`."""
        from fitwise.plotting import plot_result

        return plot_result(
            self, residuals=residuals, band=band, level=level, ax=ax, show_params=show_params
        )

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        """JSON-serializable record of the fit; see :func:`fitwise.serialize.to_dict`."""
        from fitwise.serialize import to_dict

        return to_dict(self, include_data=include_data)


def build_result(
    problem: FitProblem,
    solved: BackendResult,
    *,
    strategies: Mapping[str, str],
    scale_covar: bool | None,
    method_reason: str,
    options: Mapping[str, Any],
) -> FitResult:
    """Assemble a :class:`FitResult` from a solved problem."""
    model, data = problem.model, problem.data
    free = problem.free_names

    values, signs = model.canonicalize(problem.all_params(solved.values))
    cov_unscaled = solved.cov_unscaled
    if cov_unscaled is not None:
        d = np.array([signs[n] for n in free])
        cov_unscaled = cov_unscaled * np.outer(d, d)

    w = data.weights
    resid = data.y - np.broadcast_to(model.eval(data.x, values), data.y.shape)
    vertical_chi2 = float(np.sum((resid * w) ** 2))
    # ODR minimizes deviations in x and y; its χ² is the backend's own objective
    chi2 = solved.chi2 if solved.chi2 is not None else vertical_chi2
    rss = float(resid @ resid)
    n, k = data.n, len(free)
    ndf = n - k
    redchi = chi2 / ndf if ndf > 0 else None
    has_yerr = data.yerr is not None
    pvalue = float(stats.chi2.sf(chi2, ndf)) if has_yerr and ndf > 0 else None
    residual_sd = math.sqrt(rss / ndf) if ndf > 0 else None
    r2 = _r_squared(data.y, w, vertical_chi2)  # explained variance in y, also for ODR
    log_term = n * math.log(chi2 / n) if chi2 > 0 else -math.inf
    aic, bic = log_term + 2 * k, log_term + k * math.log(n)

    robust_weights = solved.robust_weights
    if robust_weights is not None:
        # With outliers, χ² over all points says little. The robust χ²/ndf is the squared
        # scatter of the weighted residuals estimated by 1.4826 · MAD, which outliers cannot
        # inflate; it also scales the covariance without yerr.
        rw = resid * w
        mad = float(np.median(np.abs(rw - np.median(rw))))
        redchi = (1.4826 * mad) ** 2 * n / ndf if ndf > 0 else None
        pvalue = None

    scale = (not has_yerr) if scale_covar is None else scale_covar
    log = list(solved.log)
    cov: FloatArray | None = cov_unscaled
    if cov is not None and scale:
        if redchi is None:
            cov = None
            log.append("no degrees of freedom left; cannot scale the covariance")
        else:
            cov = cov * redchi
    stderr = _stderr(cov, free)

    params = Parameters(
        [
            Parameter(
                name=name,
                value=float(values[name]),
                stderr=stderr.get(name),
                init_value=problem.fixed.get(name, problem.init.get(name, math.nan)),
                guess_strategy=strategies[name],
                bounds=problem.bounds.get(name, (-math.inf, math.inf)),
                fixed=name in problem.fixed,
            )
            for name in model.param_names
        ],
        covariance=cov,
        free_names=free,
    )
    result = FitResult(
        model=model,
        params=params,
        data=data,
        method=solved.method,
        method_reason=method_reason,
        converged=solved.converged,
        nfev=solved.nfev,
        message=solved.message,
        free_names=free,
        covariance=cov,
        scale_covar=scale,
        chi2=chi2,
        ndf=ndf,
        redchi=redchi,
        pvalue=pvalue,
        rss=rss,
        residual_sd=residual_sd,
        r2=r2,
        aic=aic,
        bic=bic,
        warnings=(),
        solver_log=tuple(log),
        raw=solved.raw,
        options=dict(options),
        loss=str(options.get("loss", "linear")),
        robust_weights=robust_weights,
    )
    return _with_warnings(result)


def _with_warnings(result: FitResult) -> FitResult:
    from dataclasses import replace

    found: list[fw_warnings.FitWarning] = []
    if not result.converged:
        found.append(
            fw_warnings.NotConvergedWarning(f"the solver did not converge: {result.message}")
        )
    if result.covariance is None and result.free_names:
        found.append(
            fw_warnings.SingularCovarianceWarning(
                "the covariance matrix could not be estimated; at least one parameter is not "
                "determined by the data (redundant parameters or too few points?)"
            )
        )
    at_bound = [p for p in result.params.values() if not p.fixed and _at_bound(p)]
    if at_bound:
        names = ", ".join(f"{p.name} = {p.value:g}" for p in at_bound)
        found.append(
            fw_warnings.ParameterAtBoundWarning(
                f"parameter(s) at a bound: {names}; their uncertainties are not reliable"
            )
        )
    robust = result.robust_weights is not None
    if result.yerr is not None and result.ndf > 0 and result.redchi is not None and not robust:
        lo, hi = stats.chi2.ppf([CHI2_ALPHA / 2, 1 - CHI2_ALPHA / 2], result.ndf)
        if result.chi2 > hi:
            found.append(
                fw_warnings.PoorChiSquareWarning(
                    f"χ²/ndf = {result.redchi:.3g} is too large for {result.ndf} degrees of "
                    f"freedom (p = {result.pvalue:.2g}): the model does not describe the data, "
                    "or yerr is underestimated"
                )
            )
        elif result.chi2 < lo:
            found.append(
                fw_warnings.PoorChiSquareWarning(
                    f"χ²/ndf = {result.redchi:.3g} is too small for {result.ndf} degrees of "
                    "freedom: yerr is probably overestimated"
                )
            )
    strong = result.correlations(STRONG_CORRELATION)
    if strong:
        pairs = ", ".join(f"{a}–{b} ({r:+.4f})" for (a, b), r in strong.items())
        found.append(
            fw_warnings.StrongCorrelationWarning(
                f"almost fully correlated parameters: {pairs}; the data determine only a "
                "combination of them. Use the covariance when combining them, or reparametrize "
                "(e.g. measure x relative to the center of the data)"
            )
        )
    large = [p for p in result.params.values() if p.stderr is not None and p.stderr > abs(p.value)]
    if large:
        names = ", ".join(p.name for p in large)
        found.append(
            fw_warnings.LargeUncertaintyWarning(
                f"relative uncertainty above 100 % for: {names}; the value is compatible with "
                "zero (consider fixing it) or poorly constrained by the data"
            )
        )
    return replace(result, warnings=tuple(found))


def _at_bound(p: Parameter) -> bool:
    for b in p.bounds:
        if math.isfinite(b):
            tol = 1e-6 * max(abs(b), abs(p.init_value)) or 1e-12
            if abs(p.value - b) <= tol:
                return True
    return False


def _stderr(cov: FloatArray | None, names: tuple[str, ...]) -> dict[str, float]:
    if cov is None:
        return {}
    diag = np.diag(cov)
    return {n: float(math.sqrt(v)) for n, v in zip(names, diag, strict=True) if v >= 0}


def _correlation(cov: FloatArray) -> FloatArray:
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    return np.asarray(np.clip(corr, -1, 1), dtype=float)


def _r_squared(y: FloatArray, w: FloatArray, chi2: float) -> float:
    w2 = w**2
    ybar = np.sum(w2 * y) / np.sum(w2)
    ss_tot = float(np.sum(w2 * (y - ybar) ** 2))
    return 1 - chi2 / ss_tot if ss_tot > 0 else math.nan
