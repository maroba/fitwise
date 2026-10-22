"""Input validation with error messages that tell the user what to do."""

from __future__ import annotations

import difflib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fitwise._typing import ArrayLike, Bounds, FloatArray

NanPolicy = Literal["raise", "omit"]


@dataclass(frozen=True)
class FitData:
    """Validated data of one fit."""

    x: FloatArray
    y: FloatArray
    yerr: FloatArray | None
    n_omitted: int = 0
    xerr: FloatArray | None = None

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def weights(self) -> FloatArray:
        """Residual weights ``1/yerr`` (ones without ``yerr``)."""
        return np.ones_like(self.y) if self.yerr is None else 1.0 / self.yerr


def validate_data(
    x: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike | None,
    nan_policy: NanPolicy,
    xerr: ArrayLike | None = None,
) -> FitData:
    """Convert to float arrays, check shapes and handle non-finite values."""
    if nan_policy not in ("raise", "omit"):
        raise ValueError(f"nan_policy must be 'raise' or 'omit', not {nan_policy!r}")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if ya.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {ya.shape}")
    if xa.ndim == 0 or xa.shape[0] != ya.shape[0]:
        raise ValueError(
            f"x and y must have the same length along the first axis, "
            f"got x.shape={xa.shape} and y.shape={ya.shape}"
        )

    ea: FloatArray | None = None
    if yerr is not None:
        ea = np.broadcast_to(np.asarray(yerr, dtype=float), ya.shape).copy()
    xe: FloatArray | None = None
    if xerr is not None:
        if xa.ndim != 1:
            raise ValueError("xerr is only supported for one-dimensional x")
        xe = np.broadcast_to(np.asarray(xerr, dtype=float), ya.shape).copy()

    finite = np.isfinite(ya) & np.isfinite(xa.reshape(len(ya), -1)).all(axis=1)
    if ea is not None:
        finite &= np.isfinite(ea)
    if xe is not None:
        finite &= np.isfinite(xe)
    n_bad = int((~finite).sum())
    if n_bad:
        if nan_policy == "raise":
            raise ValueError(
                f"{n_bad} data point(s) contain NaN or inf in x, y, yerr or xerr. "
                "Clean the data or pass nan_policy='omit' to drop them."
            )
        xa, ya = xa[finite], ya[finite]
        ea = None if ea is None else ea[finite]
        xe = None if xe is None else xe[finite]

    if ea is not None and np.any(ea <= 0):
        raise ValueError("yerr must be strictly positive; zero uncertainty gives infinite weight")
    if xe is not None and np.any(xe <= 0):
        raise ValueError("xerr must be strictly positive; use xerr=None for exact x values")
    return FitData(xa, ya, ea, n_bad, xe)


def check_names(argument: str, given: Iterable[str], known: Iterable[str]) -> None:
    """Raise a helpful error if ``given`` contains names not in ``known``."""
    known = list(known)
    for name in given:
        if name not in known:
            hint = difflib.get_close_matches(name, known, n=1)
            suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
            raise ValueError(
                f"Unknown parameter {name!r} in {argument}.{suggestion} "
                f"Model parameters: {', '.join(known)}"
            )


def as_float(argument: str, name: str, value: Any) -> float:
    """Convert a user-supplied scalar to float with a clear error."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{argument}[{name!r}] must be a number, got {value!r}") from None
    return f


def normalize_bounds(bounds: Mapping[str, Any] | None) -> dict[str, Bounds]:
    """Turn ``{name: (lo, hi)}`` with ``None`` for open sides into float pairs."""
    result: dict[str, Bounds] = {}
    for name, pair in (bounds or {}).items():
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            raise TypeError(f"bounds[{name!r}] must be a pair (lo, hi), got {pair!r}") from None
        lo_f = -math.inf if lo is None else as_float("bounds", name, lo)
        hi_f = math.inf if hi is None else as_float("bounds", name, hi)
        if math.isnan(lo_f) or math.isnan(hi_f) or not lo_f < hi_f:
            raise ValueError(f"bounds[{name!r}] = {pair!r}: need lo < hi")
        result[name] = (lo_f, hi_f)
    return result
