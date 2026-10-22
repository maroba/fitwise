"""NIST StRD linear regression reference datasets.

The 11 datasets from https://www.itl.nist.gov/div898/strd/lls/lls.shtml are vendored unchanged
in ``tests/data/nist_linear``. Model functions have the signature ``f(x, *params)`` and come
with their parameter names (``B0``, ``B1``, ...).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DATA_DIR = Path(__file__).parent / "data" / "nist_linear"
Array = NDArray[np.float64]


def _polynomial(degree: int) -> tuple[Callable[..., Array], tuple[str, ...]]:
    def f(x, *b):
        return np.polynomial.polynomial.polyval(x, b)

    return f, tuple(f"B{i}" for i in range(degree + 1))


def _no_intercept(x, b1):
    return b1 * x


def _longley(x, *b):
    return b[0] + x @ np.asarray(b[1:])


MODELS: dict[str, tuple[Callable[..., Array], tuple[str, ...]]] = {
    "Norris": _polynomial(1),
    "Pontius": _polynomial(2),
    "NoInt1": (_no_intercept, ("B1",)),
    "NoInt2": (_no_intercept, ("B1",)),
    "Filip": _polynomial(10),
    "Longley": (_longley, tuple(f"B{i}" for i in range(7))),
    **{f"Wampler{i}": _polynomial(5) for i in range(1, 6)},
}
NAMES: tuple[str, ...] = tuple(MODELS)

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PARAM_RE = re.compile(rf"^\s*(B\d+)\s+({_NUM})\s+({_NUM})\s*$")
_DATA_HEADER_RE = re.compile(r"^\s*Data:\s+y\s")


@dataclass(frozen=True)
class LinearDataset:
    """One NIST StRD linear regression problem with certified results."""

    name: str
    difficulty: str
    model: Callable[..., Array]
    param_names: tuple[str, ...]
    x: Array
    y: Array
    certified_values: Array
    certified_stderr: Array
    certified_residual_sd: float

    @property
    def dof(self) -> int:
        return len(self.y) - len(self.param_names)

    def residual_sd(self, params: Array) -> float:
        r = self.y - self.model(self.x, *params)
        return float(np.sqrt(r @ r / self.dof))


@cache
def load(name: str) -> LinearDataset:
    """Load and parse one dataset by its NIST name, e.g. ``"Filip"``."""
    if name not in MODELS:
        raise KeyError(f"unknown NIST dataset {name!r}; choose from {NAMES}")
    lines = (DATA_DIR / f"{name}.dat").read_text().splitlines()
    model, names = MODELS[name]

    difficulty = next(
        m.group(1).lower()
        for line in lines
        if (m := re.search(r"(Lower|Average|Higher) Level of Difficulty", line))
    )
    params = {
        m.group(1): (float(m.group(2)), float(m.group(3)))
        for line in lines
        if (m := _PARAM_RE.match(line))
    }
    values = np.array([params[n][0] for n in names])
    stderr = np.array([params[n][1] for n in names])
    i = next(i for i, line in enumerate(lines) if line.strip() == "Residual")
    residual_sd = float(lines[i + 1].split()[-1])

    header = next(i for i, line in enumerate(lines) if _DATA_HEADER_RE.match(line))
    table = np.array(
        [[float(v) for v in line.split()] for line in lines[header + 1 :] if line.strip()]
    )
    x = table[:, 1] if table.shape[1] == 2 else table[:, 1:]
    return LinearDataset(
        name, difficulty, model, names, x, table[:, 0], values, stderr, residual_sd
    )
