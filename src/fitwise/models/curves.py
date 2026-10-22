"""Exponential, sigmoid, power-law and oscillating models."""

from __future__ import annotations

import math

import numpy as np

from fitwise._typing import ArrayLike, FloatArray
from fitwise.model import Model
from fitwise.models._heuristics import (
    dominant_frequency,
    edge_means,
    loglinear,
    sine_projection,
    sorted_xy,
)

__all__ = ["damped_sine", "exponential", "powerlaw", "sigmoid"]


def _exponential(x: ArrayLike, amp: float, tau: float) -> FloatArray:
    return amp * np.exp(-np.asarray(x, dtype=float) / tau)


def _sigmoid(x: ArrayLike, amp: float, cen: float, width: float) -> FloatArray:
    z = -(np.asarray(x, dtype=float) - cen) / width
    return amp / (1 + np.exp(np.clip(z, -700, 700)))


def _powerlaw(x: ArrayLike, amp: float, exponent: float) -> FloatArray:
    return amp * np.asarray(x, dtype=float) ** exponent


def _damped_sine(x: ArrayLike, amp: float, freq: float, phase: float, tau: float) -> FloatArray:
    xa = np.asarray(x, dtype=float)
    return amp * np.exp(-xa / tau) * np.sin(2 * np.pi * freq * xa + phase)


def _guess_exponential(x: FloatArray, y: FloatArray) -> dict[str, float]:
    sign, a, b = loglinear(*sorted_xy(x, y))
    tau = -1 / b if b != 0 else 1e3 * (np.ptp(x) or 1.0)
    return {"amp": sign * math.exp(a), "tau": float(tau)}


def _guess_sigmoid(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = sorted_xy(x, y)
    left, right = edge_means(ys)
    rising = abs(right) >= abs(left)
    amp = right if rising else left
    i = int(np.argmin(np.abs(ys - amp / 2)))
    width = (np.ptp(xs) or 1.0) / 20
    return {"amp": amp, "cen": float(xs[i]), "width": width if rising else -width}


def _guess_powerlaw(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = sorted_xy(x, y)
    pos = xs > 0
    sign, a, b = loglinear(np.log(xs[pos]), ys[pos])
    return {"amp": sign * math.exp(a), "exponent": b}


def _guess_damped_sine(x: FloatArray, y: FloatArray) -> dict[str, float]:
    xs, ys = sorted_xy(x, y)
    freq = dominant_frequency(xs, ys)
    half = len(xs) // 2
    r1, _ = sine_projection(xs[:half], ys[:half], freq)
    r2, _ = sine_projection(xs[half:], ys[half:], freq)
    c1, c2 = float(np.mean(xs[:half])), float(np.mean(xs[half:]))
    tau = (c2 - c1) / math.log(r1 / r2) if r2 < r1 and r2 > 0 else 10 * (np.ptp(xs) or 1.0)
    # phase from the whole record after removing the estimated envelope
    _, phase = sine_projection(xs, ys * np.exp(xs / tau), freq)
    amp = r1 * math.exp(c1 / tau)
    return {"amp": amp, "freq": freq, "phase": phase, "tau": tau}


exponential = Model(
    _exponential,
    name="exponential",
    guess=_guess_exponential,
    guess_label="log-linear fit",
    linear=False,
    shape="monotone",
    abbrev="e",
    formula=r"\mathrm{amp}\,e^{-x/\tau}",
    param_docs={"amp": "value at x = 0", "tau": "decay length (negative for growth)"},
)
"""Exponential decay ``amp * exp(-x/tau)``; add :data:`constant` for an offset."""

sigmoid = Model(
    _sigmoid,
    name="sigmoid",
    guess=_guess_sigmoid,
    guess_label="step heuristic",
    linear=False,
    shape="monotone",
    abbrev="s",
    formula=r"\frac{\mathrm{amp}}{1 + e^{-(x-\mathrm{cen})/w}}",
    param_docs={
        "amp": "step height",
        "cen": "midpoint",
        "width": "transition width (negative for a falling step)",
    },
)
"""Logistic step from 0 to ``amp`` centred at ``cen``; add :data:`constant` for an offset."""

powerlaw = Model(
    _powerlaw,
    name="powerlaw",
    guess=_guess_powerlaw,
    guess_label="log-log fit",
    linear=False,
    shape="monotone",
    abbrev="pl",
    formula=r"\mathrm{amp}\,x^{k}",
    param_docs={"amp": "value at x = 1", "exponent": "power-law exponent"},
)
"""Power law ``amp * x**exponent``."""

damped_sine = Model(
    _damped_sine,
    name="damped_sine",
    guess=_guess_damped_sine,
    guess_label="periodogram",
    linear=False,
    shape="oscillation",
    abbrev="ds",
    formula=r"\mathrm{amp}\,e^{-x/\tau}\sin(2\pi f x + \varphi)",
    param_docs={
        "amp": "amplitude at x = 0",
        "freq": "frequency in cycles per unit x",
        "phase": "phase in radians",
        "tau": "decay length of the envelope",
    },
)
"""Exponentially damped sine oscillation."""
