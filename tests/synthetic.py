"""Random synthetic problems with known truth for the built-in models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import fitwise as fw
from fitwise.model import Model


@dataclass(frozen=True)
class Case:
    model: Model
    x: np.ndarray
    truth: dict[str, float]
    noise: float

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        y = self.model.eval(self.x, self.truth)
        return y + rng.normal(0, self.noise, y.shape), np.full(y.shape, self.noise)


def _sign(rng):
    return rng.choice([-1.0, 1.0])


def _peak(rng, width_name):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    return amp, {"amp": amp, "cen": rng.uniform(-3, 3), width_name: rng.uniform(0.2, 2)}


def gaussian(rng):
    amp, truth = _peak(rng, "sigma")
    return Case(fw.models.gaussian, np.linspace(-6, 6, 200), truth, 0.02 * abs(amp))


def lorentzian(rng):
    amp, truth = _peak(rng, "gamma")
    return Case(fw.models.lorentzian, np.linspace(-6, 6, 200), truth, 0.02 * abs(amp))


def voigt(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {
        "amp": amp,
        "cen": rng.uniform(-3, 3),
        "sigma": rng.uniform(0.1, 1),
        "gamma": rng.uniform(0.1, 1),
    }
    return Case(fw.models.voigt, np.linspace(-6, 6, 200), truth, 0.02 * abs(amp))


def exponential(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {"amp": amp, "tau": rng.uniform(0.5, 5)}
    return Case(fw.models.exponential, np.linspace(0, 10, 100), truth, 0.02 * abs(amp))


def sigmoid(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {"amp": amp, "cen": rng.uniform(-3, 3), "width": _sign(rng) * rng.uniform(0.1, 1)}
    return Case(fw.models.sigmoid, np.linspace(-6, 6, 120), truth, 0.02 * abs(amp))


def powerlaw(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {"amp": amp, "exponent": rng.uniform(-2, 2)}
    return Case(fw.models.powerlaw, np.linspace(0.5, 10, 80), truth, 0.01 * abs(amp))


def damped_sine(rng):
    truth = {
        "amp": rng.uniform(0.5, 5),
        "freq": rng.uniform(0.3, 2),
        "phase": rng.uniform(-np.pi, np.pi),
        "tau": rng.uniform(2, 20),
    }
    return Case(fw.models.damped_sine, np.linspace(0, 10, 400), truth, 0.02 * truth["amp"])


def linear(rng):
    truth = {"slope": rng.normal(0, 3), "intercept": rng.normal(0, 3)}
    return Case(fw.models.linear, np.linspace(-5, 5, 50), truth, 0.1)


def polynomial3(rng):
    truth = {f"c{i}": rng.normal(0, 2) for i in range(4)}
    return Case(fw.models.polynomial(3), np.linspace(-2, 2, 60), truth, 0.1)


def gaussian_linear(rng):
    amp, peak = _peak(rng, "sigma")
    peak["cen"] = rng.uniform(-2, 2)
    peak["sigma"] = rng.uniform(0.2, 1.2)
    truth = {**peak, "slope": rng.normal(0, 0.1 * abs(amp)), "intercept": rng.normal(0, abs(amp))}
    return Case(
        fw.models.gaussian + fw.models.linear, np.linspace(-6, 6, 200), truth, 0.02 * abs(amp)
    )


def exponential_constant(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {"amp": amp, "tau": rng.uniform(0.5, 3), "c": rng.normal(0, abs(amp))}
    return Case(
        fw.models.exponential + fw.models.constant, np.linspace(0, 10, 100), truth, 0.02 * abs(amp)
    )


def two_gaussians(rng):
    a1 = rng.uniform(1, 10)
    a2 = rng.uniform(0.3, 1) * a1
    truth = {
        "g1_amp": a1,
        "g1_cen": rng.uniform(-4, -1),
        "g1_sigma": rng.uniform(0.2, 0.8),
        "g2_amp": a2,
        "g2_cen": rng.uniform(1, 4),
        "g2_sigma": rng.uniform(0.2, 0.8),
    }
    return Case(fw.models.gaussian + fw.models.gaussian, np.linspace(-6, 6, 300), truth, 0.02 * a1)


def sigmoid_constant(rng):
    amp = _sign(rng) * rng.uniform(0.5, 10)
    truth = {
        "amp": amp,
        "cen": rng.uniform(-2, 2),
        "width": _sign(rng) * rng.uniform(0.1, 1),
        "c": rng.normal(0, abs(amp)),
    }
    return Case(
        fw.models.sigmoid + fw.models.constant, np.linspace(-6, 6, 120), truth, 0.02 * abs(amp)
    )


CASES: dict[str, Callable[[np.random.Generator], Case]] = {
    f.__name__: f
    for f in [
        gaussian,
        lorentzian,
        voigt,
        exponential,
        sigmoid,
        powerlaw,
        damped_sine,
        linear,
        polynomial3,
        gaussian_linear,
        exponential_constant,
        two_gaussians,
        sigmoid_constant,
    ]
}
