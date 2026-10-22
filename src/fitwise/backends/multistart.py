"""Multi-start with variable projection for parameters without starting values.

Used when some free parameters have neither ``p0``, a signature default nor a model heuristic.

1. Parameters that enter the model linearly (with the others held fixed) are found
   numerically. For any values of the remaining *nonlinear* parameters, the linear ones follow
   from a linear least-squares problem (variable projection).
2. Candidates for the nonlinear parameters without starting values are drawn from a mixture:
   characteristic values of the data (position of the dominant feature, steepest slope, width,
   dominant frequency, range and spacing of ``x``, and their reciprocals) with jitter,
   uniform positions within the range of ``x``, and log-uniform magnitudes of both signs over
   many decades. Each candidate is scored by χ² after solving for the linear parameters.
3. The best candidates are refined by the full nonlinear solver; the best result wins.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field, replace

import numpy as np

from fitwise._typing import FloatArray
from fitwise.backends.base import BackendResult, FitProblem
from fitwise.backends.least_squares import LeastSquaresBackend

#: Relative tolerance of the partial-linearity checks.
_AFFINE_RTOL = 1e-8


def conditionally_linear(problem: FitProblem, rng: np.random.Generator) -> list[str]:
    """Free parameters in which the model is jointly affine when the others are held fixed.

    The check runs at data-adapted test points. Points where a tested parameter has no effect
    at all (for example a peak far outside the data) carry no information and are skipped; at
    least two informative points are required.
    """
    names = list(problem.free_names)
    sampler = _Sampler(problem)
    points = [np.array([problem.init[n] for n in names])]
    points += [np.array([sampler.draw(rng, n) for n in names]) for _ in range(8)]

    def is_affine(subset: list[str]) -> bool:
        idx = [names.index(n) for n in subset]
        informative = 0
        for point in points:
            f0, a = _partial_design(problem, names, point, idx)
            if f0 is None or a is None or np.any(np.all(a == 0, axis=0)):
                continue
            t = rng.standard_normal(len(idx)) * np.maximum(np.abs(point[idx]), 1.0)
            moved = point.copy()
            moved[idx] = t
            actual = _eval(problem, names, moved)
            if actual is None:
                continue
            with np.errstate(all="ignore"):
                predicted = f0 + a @ t
                size = float(np.max(np.abs(f0) + np.abs(a) @ np.abs(t)))
                if not np.all(np.abs(actual - predicted) <= _AFFINE_RTOL * max(size, 1e-300)):
                    return False
            informative += 1
        return informative >= 2

    single = [n for n in names if is_affine([n])]
    # greedily build a jointly affine subset (a*b*x is affine in a and in b, but not in both)
    joint: list[str] = []
    for n in single:
        if is_affine([*joint, n]):
            joint.append(n)
    return joint


def _eval(problem: FitProblem, names: list[str], values: FloatArray) -> FloatArray | None:
    params = problem.all_params(dict(zip(names, values, strict=True)))
    with np.errstate(all="ignore"):
        out = np.broadcast_to(problem.model.eval(problem.data.x, params), problem.data.y.shape)
    return out if np.all(np.isfinite(out)) else None


def _partial_design(
    problem: FitProblem, names: list[str], point: FloatArray, idx: list[int]
) -> tuple[FloatArray | None, FloatArray | None]:
    """``f0`` with the parameters ``idx`` set to zero, and the columns ``∂f/∂p_idx``."""
    zero = point.copy()
    zero[idx] = 0.0
    f0 = _eval(problem, names, zero)
    if f0 is None:
        return None, None
    columns = []
    for i in idx:
        unit = zero.copy()
        unit[i] = 1.0
        fi = _eval(problem, names, unit)
        if fi is None:
            return None, None
        columns.append(fi - f0)
    return f0, np.column_stack(columns)


@dataclass(frozen=True)
class MultiStartBackend:
    """Screen many starting points cheaply, then refine the best ones.

    Parameters
    ----------
    sample : collection of str
        Free parameters without a starting value; only these are sampled.
    n_candidates : int
        Number of sampled starting points.
    n_refine : int
        Number of best candidates refined by the nonlinear solver.
    seed : int
        Seed of the random sampling; part of the reproducibility record.
    """

    sample: Collection[str]
    n_candidates: int = 400
    n_refine: int = 5
    seed: int = 0
    local: LeastSquaresBackend = field(default_factory=LeastSquaresBackend)

    def solve(self, problem: FitProblem) -> BackendResult:
        """Run the multi-start and return the best refined result."""
        rng = np.random.default_rng(self.seed)
        names = list(problem.free_names)
        linear = conditionally_linear(problem, rng)
        nonlinear = [n for n in names if n not in linear]
        sampled = [n for n in nonlinear if n in self.sample]
        lin_idx = [names.index(n) for n in linear]

        candidates = self._candidates(problem, sampled, rng)
        scored: list[tuple[float, FloatArray]] = []
        evals = 0
        for values in candidates:
            point = np.array([values.get(n, problem.init[n]) for n in names])
            if lin_idx:
                point, chi2 = _project(problem, names, point, lin_idx)
                evals += len(lin_idx) + 2
            else:
                chi2 = _chi2(problem, names, point)
                evals += 1
            if math.isfinite(chi2):
                scored.append((chi2, point))
        scored.sort(key=lambda item: item[0])

        log = [
            f"multi-start: {len(candidates)} candidates for {', '.join(sampled) or '–'}; "
            f"linear parameters solved by projection: {', '.join(linear) or '–'}"
        ]
        best: BackendResult | None = None
        best_chi2 = math.inf
        for rank, (_, point) in enumerate(scored[: self.n_refine]):
            start = replace(problem, init=dict(zip(names, map(float, point), strict=True)))
            try:
                solved = self.local.solve(start)
            except ValueError:
                continue
            evals += solved.nfev
            chi2 = _chi2(problem, names, np.array([solved.values[n] for n in names]))
            log.append(f"refined candidate {rank + 1}: χ² = {chi2:.10g}")
            if chi2 < best_chi2 and (solved.converged or best is None):
                best, best_chi2 = solved, chi2
        if best is None:
            best = self.local.solve(problem)
            evals += best.nfev
            log.append("no candidate could be refined; fell back to the default start")
        return replace(
            best,
            nfev=evals,
            method=f"{best.method} with multi-start",
            log=(*log, *best.log),
        )

    def _candidates(
        self, problem: FitProblem, sampled: list[str], rng: np.random.Generator
    ) -> list[dict[str, float]]:
        """Return the default start plus random draws from data-adapted ranges."""
        out: list[dict[str, float]] = [{}]
        if not sampled:
            return out
        sampler = _Sampler(problem)
        out += [{n: sampler.draw(rng, n) for n in sampled} for _ in range(self.n_candidates)]
        return out


class _Sampler:
    """Draws parameter values from a mixture adapted to the data (see module docstring)."""

    def __init__(self, problem: FitProblem) -> None:
        x = np.asarray(problem.data.x, dtype=float)
        if x.ndim != 1:
            x = x.reshape(len(problem.data.y), -1)[:, 0]
        self.x_lo, self.x_hi = float(np.min(x)), float(np.max(x))
        x_mag = max(abs(self.x_lo), abs(self.x_hi), 1e-300)
        self.decades = abs(math.log10(x_mag)) + 4  # x-scales, rates 1/x and O(1) exponents
        self.features = _features(x, problem.data.y)
        self.bounds = problem.bounds

    def draw(self, rng: np.random.Generator, name: str) -> float:
        lo, hi = self.bounds[name]
        return _draw(rng, lo, hi, self.x_lo, self.x_hi, self.decades, self.features)


def _features(x: FloatArray, y: FloatArray) -> list[float]:
    """Characteristic values of the data that parameters often correspond to."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    span = float(xs[-1] - xs[0])
    if span <= 0:
        return [1.0]
    dev = np.abs(ys - np.median(ys))
    i = int(np.argmax(dev))
    above = xs[dev >= dev[i] / 2]
    width = float(above.max() - above.min()) or span / 10
    steps = np.diff(xs)
    spacing = float(np.median(steps[steps > 0])) if np.any(steps > 0) else span / len(xs)
    slope = np.gradient(ys, xs) if len(np.unique(xs)) == len(xs) else np.zeros_like(ys)
    positions = [float(xs[i]), float(xs[int(np.argmax(np.abs(slope)))]), float(np.mean(xs))]
    scales = [width, width / 2.355, span, span / 10, spacing]
    try:
        from fitwise.models._heuristics import dominant_frequency

        freq = dominant_frequency(xs, ys)
        scales += [1 / freq]
        rates = [freq, 2 * math.pi * freq]
    except (ValueError, ZeroDivisionError):
        rates = []
    reciprocal = [1 / s for s in scales if s > 0]
    return [v for v in [*positions, *scales, *reciprocal, *rates] if math.isfinite(v) and v != 0]


def _draw(
    rng: np.random.Generator,
    lo: float,
    hi: float,
    x_lo: float,
    x_hi: float,
    decades: float,
    features: list[float],
) -> float:
    if math.isfinite(lo) and math.isfinite(hi):
        if lo > 0 and hi / lo > 100:
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        return float(rng.uniform(lo, hi))
    kind = rng.random()
    if kind < 0.4 and features:
        base = float(rng.choice(features))
        sign = 1.0 if rng.random() < 0.75 else -1.0
        value = sign * base * float(math.exp(rng.normal(0, 0.3)))  # data feature with jitter
    elif kind < 0.6 and x_hi > x_lo:
        value = float(rng.uniform(x_lo, x_hi))  # position-like parameters
    else:
        value = float(10 ** rng.uniform(-decades, decades)) * float(rng.choice([-1.0, 1.0]))
    if math.isfinite(lo) and value < lo:
        value = lo + abs(value)
    if math.isfinite(hi) and value > hi:
        value = hi - abs(value)
    return value


def _chi2(problem: FitProblem, names: list[str], point: FloatArray) -> float:
    with np.errstate(all="ignore"):
        r = problem.weighted_residual(dict(zip(names, point, strict=True)))
    value = float(r @ r)
    return value if math.isfinite(value) else math.inf


def _project(
    problem: FitProblem, names: list[str], point: FloatArray, idx: list[int]
) -> tuple[FloatArray, float]:
    """Solve for the linear parameters at fixed nonlinear ones; return the point and χ²."""
    f0, a = _partial_design(problem, names, point, idx)
    if f0 is None or a is None:
        return point, math.inf
    w = problem.data.weights
    with np.errstate(all="ignore"):
        aw = a * w[:, None]
        bw = (problem.data.y - f0) * w
        norms = np.linalg.norm(aw, axis=0)
    if not (np.all(np.isfinite(aw)) and np.all(np.isfinite(bw)) and np.all(np.isfinite(norms))):
        return point, math.inf  # candidates with overflowing model values are unusable
    norms[norms == 0] = 1.0
    try:
        t, *_ = np.linalg.lstsq(aw / norms, bw, rcond=None)
    except np.linalg.LinAlgError:
        return point, math.inf
    out = point.copy()
    lo = np.array([problem.bounds[names[i]][0] for i in idx])
    hi = np.array([problem.bounds[names[i]][1] for i in idx])
    out[idx] = np.clip(t / norms, lo, hi)
    return out, _chi2(problem, names, out)
