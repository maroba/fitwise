"""Building blocks for starting-value heuristics. All functions expect ``x`` sorted ascending."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from fitwise._typing import FloatArray


class Peak(NamedTuple):
    """Height, position and full width at half maximum of the dominant peak."""

    height: float
    position: float
    fwhm: float


def sorted_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return ``x`` and ``y`` sorted by ``x``; reject multi-dimensional ``x``."""
    if x.ndim != 1:
        raise ValueError("heuristics need one-dimensional x")
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def dominant_peak(x: FloatArray, y: FloatArray) -> Peak:
    """Find the largest positive or negative excursion from zero and its FWHM."""
    i = int(np.argmax(np.abs(y)))
    height = float(y[i])
    half = height / 2
    above = np.abs(y) >= abs(half)

    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and above[hi + 1]:
        hi += 1

    def crossing(a: int, b: int) -> float:
        # linear interpolation of the half-maximum crossing between indices a and b
        ya, yb = y[a], y[b]
        return float(x[a] + (half - ya) * (x[b] - x[a]) / (yb - ya)) if yb != ya else float(x[a])

    left = crossing(lo - 1, lo) if lo > 0 else None
    right = crossing(hi, hi + 1) if hi < len(y) - 1 else None
    if left is not None and right is not None:
        fwhm = right - left
    elif left is not None:
        fwhm = 2 * (x[i] - left)
    elif right is not None:
        fwhm = 2 * (right - x[i])
    else:
        fwhm = np.ptp(x) / 2
    if not fwhm > 0:
        fwhm = np.ptp(x) / 10 or 1.0
    return Peak(height, float(x[i]), float(fwhm))


def loglinear(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """Fit ``log|y| = a + b x`` on the points carrying the dominant sign of ``y``.

    Returns ``(sign, a, b)``. Points are weighted by ``|y|`` so that the noisy tail of a decay
    does not dominate.
    """
    sign = 1.0 if np.sum(y > 0) >= np.sum(y < 0) else -1.0
    mask = sign * y > 1e-3 * np.max(np.abs(y))
    if mask.sum() < 2:
        raise ValueError("too few points with a consistent sign")
    ay = sign * y[mask]
    b, a = np.polyfit(x[mask], np.log(ay), 1, w=np.sqrt(ay / ay.max()))
    return sign, float(a), float(b)


def dominant_frequency(x: FloatArray, y: FloatArray) -> float:
    """Frequency (cycles per unit x) of the strongest periodic component (Lomb–Scargle)."""
    from scipy.signal import lombscargle

    span = np.ptp(x)
    dx = np.diff(x)
    dx = dx[dx > 0]
    if span <= 0 or len(dx) == 0:
        raise ValueError("x must span a range")
    f_max = 0.5 / np.median(dx)
    f_min = 0.5 / span
    freqs = np.linspace(f_min, f_max, min(4000, 20 * len(x)))
    power = lombscargle(x, y - y.mean(), 2 * np.pi * freqs)
    return float(freqs[int(np.argmax(power))])


def sine_projection(x: FloatArray, y: FloatArray, freq: float) -> tuple[float, float]:
    """Amplitude and phase of the best ``R sin(2π f x + φ)`` at fixed frequency."""
    w = 2 * np.pi * freq * x
    design = np.column_stack([np.sin(w), np.cos(w)])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(math.hypot(a, b)), float(math.atan2(b, a))


def edge_means(y: FloatArray, fraction: float = 0.1) -> tuple[float, float]:
    """Mean of the first and the last ``fraction`` of the points."""
    m = max(1, int(len(y) * fraction))
    return float(np.mean(y[:m])), float(np.mean(y[-m:]))
