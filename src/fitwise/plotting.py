"""Plots of fit results (requires matplotlib: ``pip install fitwise[plot]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from fitwise.report import format_measurement
from fitwise.units import format_unit

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from fitwise.result import FitResult

__all__ = ["plot_result"]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "plotting needs matplotlib; install it with: pip install 'fitwise[plot]'"
        ) from None
    return plt


def _label(name: str, unit: Any) -> str:
    text = format_unit(unit)
    return f"{name} ({text})" if text else name


def plot_result(
    result: FitResult,
    *,
    residuals: bool = True,
    band: Literal["confidence", "prediction"] | None = "confidence",
    level: float = 0.95,
    ax: Axes | None = None,
    show_params: bool = False,
    n_points: int = 500,
) -> tuple[Figure, list[Axes]]:
    """Plot data with error bars, the fitted curve and the residuals.

    Parameters
    ----------
    result : FitResult
        The fit to show.
    residuals : bool
        Add a residual panel below the main plot (normalized by ``yerr`` if given).
        Ignored if ``ax`` is given.
    band : {"confidence", "prediction"} or None
        Shade the confidence band (where the true curve lies) or the prediction band (where
        new measurements fall). Skipped if the covariance is not available.
    level : float
        Coverage probability of the band.
    ax : matplotlib.axes.Axes, optional
        Draw into existing axes instead of creating a new figure.
    show_params : bool
        Put the fitted parameters into the legend.
    n_points : int
        Number of points of the smooth fit curve.

    Returns
    -------
    figure, axes
        The figure and the list of axes (main plot first).
    """
    if result.x.ndim != 1:
        raise ValueError("plotting is only supported for one-dimensional x")
    plt = _pyplot()
    if ax is not None:
        fig = ax.get_figure()
        axes = [ax]
    elif residuals:
        fig, (main, res) = plt.subplots(
            2, 1, sharex=True, height_ratios=(3, 1), figsize=(6.4, 5.6), layout="constrained"
        )
        axes = [main, res]
    else:
        fig, main = plt.subplots(layout="constrained")
        axes = [main]
    main = axes[0]

    x, y, yerr = result.x, result.y, result.yerr
    main.errorbar(
        x, y, yerr=yerr, xerr=result.data.xerr, fmt="o", ms=3, lw=1, capsize=0, color="C0",
        label="data", zorder=2,
    )  # fmt: skip

    out = result.downweighted
    if out.any():
        main.plot(
            x[out], y[out], "o", ms=7, mfc="none", mec="C3", mew=1.2, zorder=4,
            label=f"downweighted ({result.loss})",
        )  # fmt: skip

    xs = np.linspace(float(np.min(x)), float(np.max(x)), n_points)
    label = "fit"
    if show_params:
        rows = []
        for p in result.params.values():
            value, err = format_measurement(p.value, p.stderr)
            rows.append(f"{p.name} = {value}" + ("" if err == "–" else f" ± {err}"))
        label = "\n".join(["fit", *rows])
    if band is not None and result.covariance is not None:
        from fitwise.bands import confidence_band, describe, prediction_band

        try:
            b = (confidence_band if band == "confidence" else prediction_band)(result, xs, level)
        except ValueError:
            b = None  # e.g. prediction band with point-dependent yerr
        if b is not None:
            main.fill_between(xs, b.lower, b.upper, color="C1", alpha=0.25, lw=0, label=describe(b))
    curve = np.broadcast_to(result.eval(xs), xs.shape)
    main.plot(xs, curve, color="C1", lw=1.8, label=label, zorder=3)
    main.set_ylabel(_label("y", result.data.y_unit))
    main.legend(frameon=False)

    if len(axes) == 2:
        res_ax = axes[1]
        r = result.normalized_residuals
        res_ax.axhline(0, color="0.5", lw=0.8)
        res_ax.plot(x, r, "o", ms=3, color="C0")
        if out.any():
            res_ax.plot(x[out], r[out], "o", ms=7, mfc="none", mec="C3", mew=1.2)
        res_ax.set_ylabel("(y − f)/σ" if yerr is not None else "y − f")
        res_ax.set_xlabel(_label("x", result.data.x_unit))
    else:
        main.set_xlabel(_label("x", result.data.x_unit))
    assert fig is not None
    return fig, axes  # type: ignore[return-value]
