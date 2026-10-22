"""Peak shapes. ``amp`` is always the peak height, ``cen`` the position."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import voigt_profile

from fitwise._typing import ArrayLike, FloatArray
from fitwise.model import Model
from fitwise.models._heuristics import dominant_peak, sorted_xy

__all__ = ["gaussian", "lorentzian", "voigt"]

_FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))


def _gaussian(x: ArrayLike, amp: float, cen: float, sigma: float) -> FloatArray:
    xa = np.asarray(x, dtype=float)
    return amp * np.exp(-((xa - cen) ** 2) / (2 * sigma**2))


def _lorentzian(x: ArrayLike, amp: float, cen: float, gamma: float) -> FloatArray:
    xa = np.asarray(x, dtype=float)
    return amp * gamma**2 / ((xa - cen) ** 2 + gamma**2)


def _voigt(x: ArrayLike, amp: float, cen: float, sigma: float, gamma: float) -> FloatArray:
    xa = np.asarray(x, dtype=float)
    s, g = abs(sigma), abs(gamma)
    return np.asarray(amp * voigt_profile(xa - cen, s, g) / voigt_profile(0.0, s, g), dtype=float)


def _guess_gaussian(x: FloatArray, y: FloatArray) -> dict[str, float]:
    peak = dominant_peak(*sorted_xy(x, y))
    return {"amp": peak.height, "cen": peak.position, "sigma": peak.fwhm / _FWHM_PER_SIGMA}


def _guess_lorentzian(x: FloatArray, y: FloatArray) -> dict[str, float]:
    peak = dominant_peak(*sorted_xy(x, y))
    return {"amp": peak.height, "cen": peak.position, "gamma": peak.fwhm / 2}


def _guess_voigt(x: FloatArray, y: FloatArray) -> dict[str, float]:
    # split the width equally: FWHM ≈ 0.5346 fL + sqrt(0.2166 fL² + fG²) (Olivero & Longbothum)
    peak = dominant_peak(*sorted_xy(x, y))
    w = peak.fwhm / 1.64
    return {"amp": peak.height, "cen": peak.position, "sigma": w / _FWHM_PER_SIGMA, "gamma": w / 2}


gaussian = Model(
    _gaussian,
    name="gaussian",
    guess=_guess_gaussian,
    guess_label="peak heuristic",
    linear=False,
    shape="peak",
    symmetric=("sigma",),
    abbrev="g",
    formula=r"\mathrm{amp}\,\exp\!\left(-\frac{(x-\mathrm{cen})^2}{2\sigma^2}\right)",
    param_docs={
        "amp": "peak height",
        "cen": "peak position",
        "sigma": "standard deviation; FWHM = 2.3548 sigma",
    },
)
"""Gaussian peak of height ``amp`` at ``cen`` with standard deviation ``sigma``."""

lorentzian = Model(
    _lorentzian,
    name="lorentzian",
    guess=_guess_lorentzian,
    guess_label="peak heuristic",
    linear=False,
    shape="peak",
    symmetric=("gamma",),
    abbrev="l",
    formula=r"\mathrm{amp}\,\frac{\gamma^2}{(x-\mathrm{cen})^2 + \gamma^2}",
    param_docs={
        "amp": "peak height",
        "cen": "peak position",
        "gamma": "half width at half maximum; FWHM = 2 gamma",
    },
)
"""Lorentzian peak of height ``amp`` at ``cen`` with half width ``gamma``."""

voigt = Model(
    _voigt,
    name="voigt",
    guess=_guess_voigt,
    guess_label="peak heuristic",
    linear=False,
    shape="peak",
    symmetric=("sigma", "gamma"),
    abbrev="v",
    formula=r"\mathrm{amp}\,\frac{V(x-\mathrm{cen};\sigma,\gamma)}{V(0;\sigma,\gamma)}",
    param_docs={
        "amp": "peak height",
        "cen": "peak position",
        "sigma": "standard deviation of the Gaussian part",
        "gamma": "half width at half maximum of the Lorentzian part",
    },
)
"""Voigt peak (Gaussian ⊗ Lorentzian) of height ``amp`` at ``cen``."""
