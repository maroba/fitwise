"""Units via ``pint`` (optional: ``pip install 'fitwise[units]'``).

``x``, ``y``, ``yerr`` and ``xerr`` may be :class:`pint.Quantity` objects. fitwise fits the
magnitudes and attaches units to the parameters: built-in models declare how each parameter
scales with the units of x and y (for example ``slope`` has the unit y/x); for other models the
units can be given with ``fit(..., units={"tau": "s"})``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fitwise._typing import FloatArray

__all__ = ["format_unit", "is_quantity", "parameter_units", "split"]


def is_quantity(value: Any) -> bool:
    """Whether ``value`` is a pint quantity (without importing pint)."""
    return hasattr(value, "magnitude") and hasattr(value, "units") and hasattr(value, "to")


def split(value: Any, unit: Any = None, *, name: str) -> tuple[Any, Any]:
    """Return ``(magnitude, unit)``; convert to ``unit`` if one is given.

    Plain numbers are taken to be in ``unit`` already.
    """
    if not is_quantity(value):
        return value, unit
    if unit is None:
        return np.asarray(value.magnitude, dtype=float), value.units
    try:
        return np.asarray(value.to(unit).magnitude, dtype=float), unit
    except Exception as err:  # pint.DimensionalityError, without importing pint
        raise ValueError(f"{name} has unit {value.units:~P}, incompatible with {unit:~P}") from err


def registry_of(*units: Any) -> Any:
    """Return the pint registry the units belong to (None if there are no units)."""
    for u in units:
        if u is not None:
            return u._REGISTRY
    return None


def parameter_units(
    dims: Mapping[str, tuple[float, float] | None],
    x_unit: Any,
    y_unit: Any,
    given: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Units of the parameters from their x/y exponents ``(ex, ey)``, overridden by ``given``."""
    registry = registry_of(x_unit, y_unit)
    out: dict[str, Any] = {}
    if registry is not None:
        one = registry.dimensionless
        xu = x_unit if x_unit is not None else one
        yu = y_unit if y_unit is not None else one
        for name, exps in dims.items():
            if exps is None:
                out[name] = None
                continue
            unit = one
            for base, exponent in ((xu, exps[0]), (yu, exps[1])):
                if exponent:  # x**0 would otherwise stay in the unit as "mA⁰"
                    unit = unit * base**exponent
            out[name] = unit
    for name, unit in (given or {}).items():
        if isinstance(unit, str):
            if registry is None:
                import pint

                registry = pint.get_application_registry()  # type: ignore[no-untyped-call]
            unit = registry.Unit(unit)
        out[name] = unit
    return out


def format_unit(unit: Any) -> str:
    """Short unit text such as ``V/mA``; empty for dimensionless or unknown."""
    if unit is None:
        return ""
    text = f"{unit:~P}"
    return "" if text in ("", "dimensionless") else text


def to_magnitude(value: Any, unit: Any, *, name: str) -> Any:
    """Convert a quantity to ``unit`` (plain numbers pass through)."""
    if not is_quantity(value):
        return value
    if unit is None:
        raise ValueError(
            f"{name} is given with a unit, but the unit of that parameter is not known; "
            "declare it with fit(..., units={...})"
        )
    magnitude, _ = split(value, unit, name=name)
    return float(magnitude) if np.ndim(magnitude) == 0 else magnitude


def attach(values: FloatArray, unit: Any) -> Any:
    """Return ``values`` as a quantity of ``unit`` (unchanged if ``unit`` is None)."""
    return values if unit is None else values * unit
