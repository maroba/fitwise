"""Benchmark catalog: realistic user models as plain functions without starting values.

Each case draws random true parameters from a typical range, simulates data with known noise,
and is fitted with ``fw.fit(func, x, y, yerr=...)`` and nothing else. Used by the slow test in
``test_free_start.py`` and for measuring convergence rates during development.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
Rng = np.random.Generator


@dataclass(frozen=True)
class FreeCase:
    name: str
    func: Callable[..., Array]
    x: Callable[[Rng], Array]
    truth: Callable[[Rng], dict[str, float]]
    noise: float = 0.01
    """Noise as a fraction of the peak-to-peak range of the true curve."""

    def sample(self, rng: Rng) -> tuple[Array, Array, Array, dict[str, float]]:
        x = self.x(rng)
        truth = self.truth(rng)
        y_true = self.func(x, **truth)
        sigma = self.noise * np.ptp(y_true) + 1e-300
        y = y_true + rng.normal(0, sigma, y_true.shape)
        return x, y, np.full_like(y, sigma), truth


def u(rng: Rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def logu(rng: Rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sign(rng: Rng) -> float:
    return float(rng.choice([-1.0, 1.0]))


def lin(lo: float, hi: float, n: int = 100) -> Callable[[Rng], Array]:
    return lambda rng: np.linspace(lo, hi, n)


# --- model functions (no default values on purpose) -----------------------------------------


def exp_decay(x, a, tau, c):
    return a * np.exp(-x / tau) + c


def exp_growth(x, a, k):
    return a * np.exp(k * x)


def double_exp(x, a1, t1, a2, t2, c):
    return a1 * np.exp(-x / t1) + a2 * np.exp(-x / t2) + c


def gauss_bg(x, a, mu, s, c):
    return a * np.exp(-((x - mu) ** 2) / (2 * s**2)) + c


def lorentz_bg(x, a, x0, g, c):
    return a * g**2 / ((x - x0) ** 2 + g**2) + c


def two_gauss(x, a1, m1, s1, a2, m2, s2):
    return a1 * np.exp(-((x - m1) ** 2) / (2 * s1**2)) + a2 * np.exp(-((x - m2) ** 2) / (2 * s2**2))


def gauss_slope(x, a, mu, s, k, c):
    return a * np.exp(-((x - mu) ** 2) / (2 * s**2)) + k * x + c


def logistic(x, top, k, x0, bottom):
    return bottom + top / (1 + np.exp(-k * (x - x0)))


def hill(x, vmax, kd, n):
    return vmax * x**n / (kd**n + x**n)


def michaelis(x, vmax, km):
    return vmax * x / (km + x)


def power(x, a, b):
    return a * x**b


def power_offset(x, a, b, c):
    return a * x**b + c


def arrhenius(x, A, Ea):
    return A * np.exp(-Ea / (8.314 * x))


def damped(x, a, tau, f, phi, c):
    return a * np.exp(-x / tau) * np.sin(2 * np.pi * f * x + phi) + c


def sine(x, a, w, phi, c):
    return a * np.sin(w * x + phi) + c


def logistic_growth(x, K, N0, r):
    return K / (1 + (K - N0) / N0 * np.exp(-r * x))


def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


def weibull_cdf(x, lam, k):
    return 1 - np.exp(-((x / lam) ** k))


def lognormal_peak(x, a, mu, s):
    return a * np.exp(-((np.log(x) - mu) ** 2) / (2 * s**2))


def planck_like(x, a, b):
    return a / (x**5 * (np.exp(b / x) - 1))


def curie_weiss(x, C, theta, chi0):
    return C / (x - theta) + chi0


def rc_charge(x, v0, tau):
    return v0 * (1 - np.exp(-x / tau))


def stretched(x, a, tau, beta):
    return a * np.exp(-((x / tau) ** beta))


def lennard_jones(x, eps, sig):
    return 4 * eps * ((sig / x) ** 12 - (sig / x) ** 6)


def morse(x, D, a, r0):
    return D * (1 - np.exp(-a * (x - r0))) ** 2


def rational(x, a, b, c):
    return (a + b * x) / (1 + c * x)


def langmuir(x, qmax, K):
    return qmax * K * x / (1 + K * x)


def freundlich(x, K, n):
    return K * x ** (1 / n)


def shockley(x, Is, n):
    return Is * (np.exp(x / (n * 0.025852)) - 1)


def malus(x, i0, th0, c):
    return i0 * np.cos(x - th0) ** 2 + c


def resonance(x, A, f0, gamma):
    return A / np.sqrt((f0**2 - x**2) ** 2 + (gamma * x) ** 2)


def log_model(x, a, b, c):
    return a + b * np.log(x - c)


CASES: list[FreeCase] = [
    FreeCase(
        "exp_decay",
        exp_decay,
        lin(0, 10),
        lambda r: {"a": sign(r) * u(r, 1, 10), "tau": u(r, 0.5, 4), "c": u(r, -2, 2)},
    ),
    FreeCase(
        "exp_decay_timestamps",
        exp_decay,
        lin(0, 1000),
        lambda r: {"a": u(r, 100, 1000), "tau": u(r, 50, 400), "c": u(r, 0, 50)},
    ),
    FreeCase("exp_growth", exp_growth, lin(0, 5), lambda r: {"a": u(r, 0.5, 5), "k": u(r, 0.2, 1)}),
    FreeCase(
        "double_exp",
        double_exp,
        lin(0, 20, 200),
        lambda r: {
            "a1": u(r, 2, 10),
            "t1": u(r, 0.3, 1),
            "a2": u(r, 1, 5),
            "t2": u(r, 3, 8),
            "c": u(r, 0, 1),
        },
        0.003,
    ),
    FreeCase(
        "gauss_wavelength",
        gauss_bg,
        lin(400, 700, 150),
        lambda r: {"a": u(r, 1, 10), "mu": u(r, 480, 620), "s": u(r, 5, 30), "c": u(r, 0, 2)},
    ),
    FreeCase(
        "lorentz_bg",
        lorentz_bg,
        lin(-10, 10, 150),
        lambda r: {"a": u(r, 1, 5), "x0": u(r, -5, 5), "g": u(r, 0.3, 2), "c": u(r, -1, 1)},
    ),
    FreeCase(
        "two_gauss",
        two_gauss,
        lin(0, 10, 200),
        lambda r: {
            "a1": u(r, 2, 5),
            "m1": u(r, 2, 4),
            "s1": u(r, 0.3, 0.6),
            "a2": u(r, 1, 3),
            "m2": u(r, 6, 8),
            "s2": u(r, 0.3, 0.6),
        },
    ),
    FreeCase(
        "gauss_slope",
        gauss_slope,
        lin(0, 100, 150),
        lambda r: {
            "a": u(r, 5, 20),
            "mu": u(r, 30, 70),
            "s": u(r, 3, 10),
            "k": u(r, -0.05, 0.05),
            "c": u(r, 0, 5),
        },
    ),
    FreeCase(
        "logistic",
        logistic,
        lin(0, 50),
        lambda r: {
            "top": u(r, 2, 10),
            "k": sign(r) * u(r, 0.2, 1),
            "x0": u(r, 15, 35),
            "bottom": u(r, 0, 2),
        },
    ),
    FreeCase(
        "hill",
        hill,
        lambda r: np.geomspace(0.01, 100, 40),
        lambda r: {"vmax": u(r, 1, 10), "kd": logu(r, 0.1, 10), "n": u(r, 0.7, 3)},
    ),
    FreeCase(
        "michaelis",
        michaelis,
        lin(0.1, 50, 30),
        lambda r: {"vmax": u(r, 1, 100), "km": u(r, 1, 20)},
    ),
    FreeCase(
        "power", power, lin(1, 100, 50), lambda r: {"a": logu(r, 0.01, 100), "b": u(r, -2, 2)}
    ),
    FreeCase(
        "power_offset",
        power_offset,
        lin(1, 50, 50),
        lambda r: {"a": u(r, 0.5, 5), "b": u(r, 0.5, 2), "c": u(r, -5, 5)},
    ),
    FreeCase(
        "arrhenius",
        arrhenius,
        lin(280, 380, 12),
        lambda r: {"A": logu(r, 1e8, 1e13), "Ea": u(r, 4e4, 9e4)},
    ),
    FreeCase(
        "damped_sine",
        damped,
        lin(0, 20, 400),
        lambda r: {
            "a": u(r, 1, 5),
            "tau": u(r, 5, 30),
            "f": u(r, 0.2, 2),
            "phi": u(r, -3, 3),
            "c": u(r, -1, 1),
        },
    ),
    FreeCase(
        "sine",
        sine,
        lin(0, 10, 200),
        lambda r: {"a": u(r, 1, 5), "w": u(r, 1, 10), "phi": u(r, -3, 3), "c": u(r, -2, 2)},
    ),
    FreeCase(
        "logistic_growth",
        logistic_growth,
        lin(0, 48, 40),
        lambda r: {"K": u(r, 1e3, 1e5), "N0": u(r, 10, 100), "r": u(r, 0.2, 0.6)},
    ),
    FreeCase(
        "gompertz",
        gompertz,
        lin(0, 30, 50),
        lambda r: {"a": u(r, 1, 10), "b": u(r, 2, 8), "c": u(r, 0.1, 0.5)},
    ),
    FreeCase(
        "weibull_cdf",
        weibull_cdf,
        lin(0.1, 300, 60),
        lambda r: {"lam": u(r, 50, 150), "k": u(r, 0.8, 4)},
    ),
    FreeCase(
        "lognormal_peak",
        lognormal_peak,
        lin(0.1, 50, 100),
        lambda r: {"a": u(r, 1, 10), "mu": u(r, 1, 2.5), "s": u(r, 0.2, 0.6)},
    ),
    FreeCase(
        "planck_like",
        planck_like,
        lin(200, 3000, 80),
        lambda r: {"a": logu(r, 1e14, 1e16), "b": u(r, 2000, 6000)},
    ),
    FreeCase(
        "curie_weiss",
        curie_weiss,
        lin(100, 300, 40),
        lambda r: {"C": u(r, 1, 10), "theta": u(r, -50, 80), "chi0": u(r, 0, 0.01)},
    ),
    FreeCase(
        "rc_charge",
        rc_charge,
        lin(0, 5e-3, 60),
        lambda r: {"v0": u(r, 1, 12), "tau": u(r, 2e-4, 2e-3)},
    ),
    FreeCase(
        "stretched_exp",
        stretched,
        lin(0.01, 10, 80),
        lambda r: {"a": u(r, 1, 5), "tau": u(r, 0.5, 3), "beta": u(r, 0.4, 1)},
    ),
    FreeCase(
        "lennard_jones",
        lennard_jones,
        lambda r: np.linspace(0.95, 3, 60),
        lambda r: {"eps": u(r, 0.5, 2), "sig": u(r, 0.9, 1.0)},
        0.002,
    ),
    FreeCase(
        "morse",
        morse,
        lin(0.7, 4, 60),
        lambda r: {"D": u(r, 2, 6), "a": u(r, 1, 2.5), "r0": u(r, 1, 1.6)},
    ),
    FreeCase(
        "rational",
        rational,
        lin(0, 10, 50),
        lambda r: {"a": u(r, 1, 5), "b": u(r, 0.5, 3), "c": u(r, 0.1, 1)},
    ),
    FreeCase(
        "langmuir",
        langmuir,
        lin(0.05, 20, 30),
        lambda r: {"qmax": u(r, 1, 50), "K": logu(r, 0.05, 5)},
    ),
    FreeCase(
        "freundlich", freundlich, lin(0.1, 50, 30), lambda r: {"K": u(r, 1, 10), "n": u(r, 1.2, 4)}
    ),
    FreeCase(
        "shockley",
        shockley,
        lin(0, 0.7, 50),
        lambda r: {"Is": logu(r, 1e-14, 1e-10), "n": u(r, 1, 2)},
    ),
    FreeCase(
        "malus",
        malus,
        lambda r: np.linspace(0, np.pi, 40),
        lambda r: {"i0": u(r, 1, 10), "th0": u(r, 0.2, 2.8), "c": u(r, 0, 0.5)},
    ),
    FreeCase(
        "resonance",
        resonance,
        lin(500, 1500, 120),
        lambda r: {"A": u(r, 1e5, 1e6), "f0": u(r, 800, 1200), "gamma": u(r, 20, 100)},
    ),
    FreeCase(
        "log_model",
        log_model,
        lin(1, 100, 50),
        lambda r: {"a": u(r, -5, 5), "b": u(r, 0.5, 3), "c": u(r, -5, 0.5)},
    ),
]
