"""Fitted parameters with uncertainties and provenance."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from fitwise._typing import Bounds, FloatArray

__all__ = ["Parameter", "Parameters"]

_UNBOUNDED: Bounds = (-math.inf, math.inf)


@dataclass(frozen=True)
class Parameter:
    """A single fitted parameter.

    Attributes
    ----------
    name : str
        Parameter name as used by the model.
    value : float
        Best-fit value (or the fixed value).
    stderr : float or None
        One-sigma standard error; ``None`` if fixed or if the covariance is unavailable.
    init_value : float
        Starting value of the fit.
    guess_strategy : str
        How the starting value was obtained, e.g. ``"p0"`` or ``"peak heuristic"``.
    bounds : tuple of float
        Lower and upper bound; ``±inf`` means unbounded.
    fixed : bool
        Whether the parameter was held fixed.
    """

    name: str
    value: float
    stderr: float | None
    init_value: float
    guess_strategy: str
    bounds: Bounds = _UNBOUNDED
    fixed: bool = False

    @property
    def relative_error(self) -> float | None:
        """``stderr / |value|``, or ``None`` if undefined."""
        if self.stderr is None or self.value == 0:
            return None
        return self.stderr / abs(self.value)

    @property
    def is_bounded(self) -> bool:
        """Whether at least one bound is finite."""
        return math.isfinite(self.bounds[0]) or math.isfinite(self.bounds[1])

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        if self.fixed:
            return f"Parameter({self.name!r}, value={self.value:g}, fixed)"
        err = "None" if self.stderr is None else f"{self.stderr:g}"
        return f"Parameter({self.name!r}, value={self.value:g}, stderr={err})"


class Parameters(Mapping[str, Parameter]):
    """Ordered, read-only collection of :class:`Parameter` objects.

    Supports item access (``params["sigma"]``) and attribute access (``params.sigma``).
    """

    def __init__(
        self,
        params: list[Parameter],
        *,
        covariance: FloatArray | None = None,
        free_names: tuple[str, ...] = (),
    ) -> None:
        self._params = {p.name: p for p in params}
        self._covariance = covariance
        self._free_names = free_names

    def __getitem__(self, name: str) -> Parameter:
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(f"no parameter {name!r}; available: {', '.join(self._params)}") from None

    def __getattr__(self, name: str) -> Parameter:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._params[name]
        except KeyError:
            raise AttributeError(f"no parameter {name!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self._params]

    def __repr__(self) -> str:
        return f"Parameters({list(self._params.values())!r})"

    def values_dict(self) -> dict[str, float]:
        """Return the best-fit values as ``{name: value}``."""
        return {n: p.value for n, p in self._params.items()}

    def errors_dict(self) -> dict[str, float | None]:
        """Return the standard errors as ``{name: stderr}``."""
        return {n: p.stderr for n, p in self._params.items()}

    def as_ufloat(self) -> dict[str, Any]:
        """Return the parameters as correlated ``uncertainties.ufloat`` values.

        Calculations with the returned values propagate errors including the correlations
        between parameters. Fixed parameters are returned as plain floats. Requires the
        ``uncertainties`` package (``pip install 'fitwise[units]'``).

        Examples
        --------
        >>> import fitwise as fw
        >>> x = np.linspace(0, 1, 20)
        >>> r = fw.fit(fw.models.linear, x, 2 * x + 1 + 0.01 * np.sin(17 * x))
        >>> u = r.params.as_ufloat()
        >>> round((u["slope"] + u["intercept"]).nominal_value, 2)
        3.0
        """
        try:
            from uncertainties import correlated_values
        except ImportError:
            raise ImportError(
                "as_ufloat() needs the uncertainties package: pip install 'fitwise[units]'"
            ) from None
        if self._covariance is None:
            raise ValueError("the covariance matrix is not available for this fit")
        values = [self._params[n].value for n in self._free_names]
        correlated = dict(
            zip(self._free_names, correlated_values(values, self._covariance), strict=True)
        )
        return {n: correlated.get(n, p.value) for n, p in self._params.items()}

    def as_array(self) -> FloatArray:
        """Return the best-fit values as an array in model order."""
        return np.array([p.value for p in self._params.values()], dtype=float)
