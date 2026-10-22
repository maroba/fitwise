"""Human-readable reports of fit results."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fitwise.units import format_unit

if TYPE_CHECKING:
    from fitwise.result import FitResult

__all__ = ["format_measurement", "text_report"]

#: Correlations above this threshold are listed in the report.
REPORT_CORRELATION = 0.5
_LABEL = 12


def format_measurement(value: float, error: float | None) -> tuple[str, str]:
    """Round a value and its uncertainty consistently.

    The uncertainty is rounded to two significant digits and the value to the same decimal
    place. Very large or small numbers use a shared power of ten.

    Examples
    --------
    >>> format_measurement(1.51234, 0.01937)
    ('1.512', '0.019')
    >>> format_measurement(56789.0, 1234.0)
    ('56800', '1200')
    >>> format_measurement(-1.42626e-6, 5.567e-8)
    ('-1.426e-06', '0.056e-06')
    >>> format_measurement(2.5, None)
    ('2.5', '–')
    """
    if error is None or not math.isfinite(error) or error <= 0 or not math.isfinite(value):
        return f"{value:.6g}", "–"
    e_exp = math.floor(math.log10(error))
    decimals = 1 - e_exp
    v_exp = math.floor(math.log10(abs(value))) if value != 0 else e_exp
    if decimals <= 6 and v_exp < 6:
        if decimals >= 0:
            return f"{value:.{decimals}f}", f"{error:.{decimals}f}"
        q = 10.0 ** (-decimals)
        return f"{round(value / q) * q:.0f}", f"{round(error / q) * q:.0f}"
    exp = max(v_exp, e_exp)
    mant = max(0, exp - e_exp + 1)
    scale = 10.0**exp
    return f"{value / scale:.{mant}f}e{exp:+03d}", f"{error / scale:.{mant}f}e{exp:+03d}"


def _format_bound(b: float) -> str:
    if math.isinf(b):
        return "∞" if b > 0 else "-∞"
    return f"{b:g}"


def _r2_text(r2: float) -> str:
    if not math.isfinite(r2):
        return "–"
    if r2 >= 1:
        return f"{r2:.4f}"
    # two significant digits of 1 - R²
    digits = min(10, max(4, 1 - math.floor(math.log10(1 - r2))))
    return f"{r2:.{digits}f}"


def _data_line(result: FitResult) -> str:
    n = f"{result.data.n} points"
    if result.data.n_omitted:
        n += f" ({result.data.n_omitted} non-finite omitted)"
    if result.yerr is not None:
        how = "scaled by χ²/ndf" if result.scale_covar else "absolute"
        given = "xerr and yerr" if result.data.xerr is not None else "yerr"
        return f"{n}, {given} given ({how})"
    if result.scale_covar:
        scatter = (
            "robust residual scatter" if result.robust_weights is not None else "residual scatter"
        )
        return f"{n}, no yerr (uncertainties from {scatter})"
    return f"{n}, no yerr (unit weights, covariance not scaled)"


def _start_line(result: FitResult) -> str:
    groups: dict[str, list[str]] = {}
    for name, strategy in result.guess_strategy.items():
        groups.setdefault(strategy, []).append(name)
    return " · ".join(f"{', '.join(names)}: {strategy}" for strategy, names in groups.items())


def _parameter_table(result: FitResult) -> list[str]:
    header = ("Parameter", "Value", "± Std.Err", "Unit", "Rel.", "Bounds")
    rows = []
    for p in result.params.values():
        value, err = format_measurement(p.value, p.stderr)
        if p.fixed:
            err, bounds = "–", "fixed"
        elif p.is_bounded:
            bounds = f"({_format_bound(p.bounds[0])}, {_format_bound(p.bounds[1])})"
        else:
            bounds = "–"
        rel = p.relative_error
        rel_text = f"{100 * rel:.1f} %" if rel is not None and not p.fixed else "–"
        rows.append((p.name, value, err, format_unit(p.unit), rel_text, bounds))
    show_units = any(r[3] for r in rows)
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]

    def line(cells: tuple[str, ...]) -> str:
        name, value, err, unit, rel, bounds = cells
        unit_cell = f"{unit:<{widths[3]}}   " if show_units else ""
        return (
            f"{name:<{widths[0]}}   {value:>{widths[1]}}   {err:>{widths[2]}}   "
            f"{unit_cell}{rel:>{widths[4]}}   {bounds}"
        ).rstrip()

    return [line(header), *(line(r) for r in rows)]


def _statistics_line(result: FitResult) -> str:
    parts = []
    if result.robust_weights is not None and result.redchi is not None:
        if result.yerr is not None:
            parts.append(f"χ²/ndf (robust) = {result.redchi:.3g}")
        else:
            parts.append(f"σ_res (robust) = {math.sqrt(result.redchi):.4g}")
    elif result.yerr is not None and result.redchi is not None:
        p = f", p = {result.pvalue:.2g}" if result.pvalue is not None else ""
        parts.append(f"χ²/ndf = {result.redchi:.3g} ({result.ndf} dof{p})")
    elif result.residual_sd is not None:
        parts.append(f"σ_res = {result.residual_sd:.4g} ({result.ndf} dof)")
    else:
        parts.append("0 dof")
    parts.append(f"R² = {_r2_text(result.r2)}")
    parts.append(f"AIC = {result.aic:.1f}")
    return "    ".join(parts)


def text_report(result: FitResult) -> str:
    """Build the plain-text report printed by ``print(result)``."""
    converged = (
        f"yes ({result.nfev} evals)"
        if result.converged
        else f"NO ({result.nfev} evals): {result.message}"
    )
    lines = [
        f"{'Model:':<{_LABEL}}{result.model.signature()}",
        f"{'Method:':<{_LABEL}}{result.method} ({result.method_reason})",
        f"{'Converged:':<{_LABEL}}{converged}",
        f"{'Data:':<{_LABEL}}{_data_line(result)}",
        f"{'Start:':<{_LABEL}}{_start_line(result)}",
        "",
        *_parameter_table(result),
        "",
        _statistics_line(result),
    ]
    if result.robust_weights is not None:
        n_down = int(result.downweighted.sum())
        lines.append(
            f"{'Robust:':<{_LABEL}}{result.loss} loss; {n_down} of {result.data.n} points "
            "downweighted (weight < 0.5)"
        )
    corr = result.correlations(REPORT_CORRELATION)
    if corr:
        pairs = ", ".join(f"{a}–{b} ({r:+.2f})" for (a, b), r in corr.items())
        lines.append(f"Correlations > {REPORT_CORRELATION}:  {pairs}")
    if result.warnings:
        first, *rest = (str(w) for w in result.warnings)
        lines.append(f"{'Warnings:':<{_LABEL}}! {first}")
        lines.extend(f"{'':<{_LABEL}}! {w}" for w in rest)
    else:
        lines.append(f"{'Warnings:':<{_LABEL}}none")
    return "\n".join(lines)
