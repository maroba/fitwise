"""NIST StRD nonlinear regression reference datasets.

The 27 datasets from https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml are vendored
unchanged in ``tests/data/nist``. This module parses them and provides the model functions,
which NIST gives only as text in the file header.

Conventions
-----------
- Model functions have the signature ``f(x, b1, b2, ...)``, like user models in fitwise.
- ``Nelson`` models ``log(y)``; its ``y`` is therefore already log-transformed, and its
  ``x`` has shape ``(n, 2)``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DATA_DIR = Path(__file__).parent / "data" / "nist"

Difficulty = Literal["lower", "average", "higher"]
Array = NDArray[np.float64]


def _misra1a(x, b1, b2):
    return b1 * (1 - np.exp(-b2 * x))


def _chwirut(x, b1, b2, b3):
    return np.exp(-b1 * x) / (b2 + b3 * x)


def _lanczos(x, b1, b2, b3, b4, b5, b6):
    return b1 * np.exp(-b2 * x) + b3 * np.exp(-b4 * x) + b5 * np.exp(-b6 * x)


def _gauss(x, b1, b2, b3, b4, b5, b6, b7, b8):
    return (
        b1 * np.exp(-b2 * x)
        + b3 * np.exp(-((x - b4) ** 2) / b5**2)
        + b6 * np.exp(-((x - b7) ** 2) / b8**2)
    )


def _danwood(x, b1, b2):
    return b1 * x**b2


def _misra1b(x, b1, b2):
    return b1 * (1 - (1 + b2 * x / 2) ** (-2))


def _kirby2(x, b1, b2, b3, b4, b5):
    return (b1 + b2 * x + b3 * x**2) / (1 + b4 * x + b5 * x**2)


def _rational_33(x, b1, b2, b3, b4, b5, b6, b7):
    return (b1 + b2 * x + b3 * x**2 + b4 * x**3) / (1 + b5 * x + b6 * x**2 + b7 * x**3)


def _nelson(x, b1, b2, b3):
    x1, x2 = x[:, 0], x[:, 1]
    return b1 - b2 * x1 * np.exp(-b3 * x2)


def _mgh17(x, b1, b2, b3, b4, b5):
    return b1 + b2 * np.exp(-x * b4) + b3 * np.exp(-x * b5)


def _misra1c(x, b1, b2):
    return b1 * (1 - (1 + 2 * b2 * x) ** (-0.5))


def _misra1d(x, b1, b2):
    return b1 * b2 * x / (1 + b2 * x)


def _roszman1(x, b1, b2, b3, b4):
    return b1 - b2 * x - np.arctan(b3 / (x - b4)) / np.pi


def _enso(x, b1, b2, b3, b4, b5, b6, b7, b8, b9):
    w = 2 * np.pi * x
    return (
        b1
        + b2 * np.cos(w / 12)
        + b3 * np.sin(w / 12)
        + b5 * np.cos(w / b4)
        + b6 * np.sin(w / b4)
        + b8 * np.cos(w / b7)
        + b9 * np.sin(w / b7)
    )


def _mgh09(x, b1, b2, b3, b4):
    return b1 * (x**2 + x * b2) / (x**2 + x * b3 + b4)


def _mgh10(x, b1, b2, b3):
    return b1 * np.exp(b2 / (x + b3))


def _rat42(x, b1, b2, b3):
    return b1 / (1 + np.exp(b2 - b3 * x))


def _eckerle4(x, b1, b2, b3):
    return (b1 / b2) * np.exp(-0.5 * ((x - b3) / b2) ** 2)


def _rat43(x, b1, b2, b3, b4):
    return b1 / (1 + np.exp(b2 - b3 * x)) ** (1 / b4)


def _bennett5(x, b1, b2, b3):
    return b1 * (b2 + x) ** (-1 / b3)


MODELS: dict[str, Callable[..., Array]] = {
    # lower difficulty
    "Misra1a": _misra1a,
    "Chwirut2": _chwirut,
    "Chwirut1": _chwirut,
    "Lanczos3": _lanczos,
    "Gauss1": _gauss,
    "Gauss2": _gauss,
    "DanWood": _danwood,
    "Misra1b": _misra1b,
    # average difficulty
    "Kirby2": _kirby2,
    "Hahn1": _rational_33,
    "Nelson": _nelson,
    "MGH17": _mgh17,
    "Lanczos1": _lanczos,
    "Lanczos2": _lanczos,
    "Gauss3": _gauss,
    "Misra1c": _misra1c,
    "Misra1d": _misra1d,
    "Roszman1": _roszman1,
    "ENSO": _enso,
    # higher difficulty
    "MGH09": _mgh09,
    "Thurber": _rational_33,
    "BoxBOD": _misra1a,
    "Rat42": _rat42,
    "MGH10": _mgh10,
    "Eckerle4": _eckerle4,
    "Rat43": _rat43,
    "Bennett5": _bennett5,
}

NAMES: tuple[str, ...] = tuple(MODELS)


@dataclass(frozen=True)
class NistDataset:
    """One NIST StRD nonlinear regression problem with certified results."""

    name: str
    difficulty: Difficulty
    model: Callable[..., Array]
    x: Array
    y: Array
    param_names: tuple[str, ...]
    start1: Array
    start2: Array
    certified_values: Array
    certified_stderr: Array
    certified_rss: float
    certified_residual_sd: float
    dof_stated: int
    """Degrees of freedom as printed in the file (wrong for Rat43, see tests)."""

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def dof(self) -> int:
        return self.n_obs - len(self.param_names)

    def rss(self, params: Array) -> float:
        """Residual sum of squares at ``params``."""
        r = self.y - self.model(self.x, *params)
        return float(r @ r)


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PARAM_RE = re.compile(rf"^\s*(b\d+)\s*=\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$")
_DIFFICULTY_RE = re.compile(r"(Lower|Average|Higher) Level of Difficulty")
_DATA_HEADER_RE = re.compile(r"^\s*Data:\s+y\s")


def _float_after(label: str, lines: list[str]) -> float:
    for line in lines:
        if line.strip().startswith(label):
            return float(line.split()[-1])
    raise ValueError(f"{label!r} not found")


@cache
def load(name: str) -> NistDataset:
    """Load and parse one dataset by its NIST name, e.g. ``"Misra1a"``."""
    if name not in MODELS:
        raise KeyError(f"unknown NIST dataset {name!r}; choose from {NAMES}")
    lines = (DATA_DIR / f"{name}.dat").read_text().splitlines()

    match = next(m for line in lines if (m := _DIFFICULTY_RE.search(line)))
    difficulty: Difficulty = match.group(1).lower()  # type: ignore[assignment]

    params = [m.groups() for line in lines if (m := _PARAM_RE.match(line))]
    names = tuple(p[0] for p in params)
    start1, start2, values, stderr = (np.array([float(p[i]) for p in params]) for i in range(1, 5))

    header = next(i for i, line in enumerate(lines) if _DATA_HEADER_RE.match(line))
    table = np.array(
        [[float(v) for v in line.split()] for line in lines[header + 1 :] if line.strip()]
    )
    y = table[:, 0]
    x = table[:, 1] if table.shape[1] == 2 else table[:, 1:]
    if name == "Nelson":
        y = np.log(y)

    return NistDataset(
        name=name,
        difficulty=difficulty,
        model=MODELS[name],
        x=x,
        y=y,
        param_names=names,
        start1=start1,
        start2=start2,
        certified_values=values,
        certified_stderr=stderr,
        certified_rss=_float_after("Residual Sum of Squares:", lines),
        certified_residual_sd=_float_after("Residual Standard Deviation:", lines),
        dof_stated=int(_float_after("Degrees of Freedom:", lines)),
    )


def load_all(difficulty: Difficulty | None = None) -> list[NistDataset]:
    """Load all datasets, optionally filtered by difficulty."""
    datasets = [load(name) for name in NAMES]
    return [d for d in datasets if difficulty is None or d.difficulty == difficulty]
