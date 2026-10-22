"""The contract between :func:`fitwise.fit` and the solver backends.

Every backend receives a normalized :class:`FitProblem` and returns a :class:`BackendResult`.
:class:`fitwise.FitResult` is built only from these two objects, so all solvers produce
results with identical structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from fitwise._typing import Bounds, FloatArray
from fitwise._validate import FitData
from fitwise.model import Model


@dataclass(frozen=True)
class FitProblem:
    """A fully specified least-squares problem.

    Attributes
    ----------
    model, data
        The model and the validated data.
    init : dict
        Starting values of the free parameters, in model order.
    fixed : dict
        Values of the fixed parameters.
    bounds : dict
        Bounds of the free parameters (``±inf`` if unbounded).
    """

    model: Model
    data: FitData
    init: dict[str, float]
    fixed: dict[str, float]
    bounds: dict[str, Bounds]

    @property
    def free_names(self) -> tuple[str, ...]:
        """Names of the varied parameters in model order."""
        return tuple(self.init)

    @property
    def has_bounds(self) -> bool:
        """Whether any free parameter has a finite bound."""
        return any(np.isfinite(b).any() for b in self.bounds.values())

    def all_params(self, free: dict[str, float]) -> dict[str, float]:
        """Merge free and fixed values into a full parameter dict in model order."""
        merged = {**free, **self.fixed}
        return {n: merged[n] for n in self.model.param_names}

    def weighted_residual(self, free: dict[str, float]) -> FloatArray:
        """``(y - f(x)) / yerr`` (or unweighted without ``yerr``)."""
        model_y = self.model.eval(self.data.x, self.all_params(free))
        return (self.data.y - np.broadcast_to(model_y, self.data.y.shape)) * self.data.weights


@dataclass(frozen=True)
class BackendResult:
    """What a backend reports back.

    Attributes
    ----------
    values : dict
        Best-fit values of the free parameters.
    cov_unscaled : ndarray or None
        ``(JᵀJ)⁻¹`` of the weighted residuals, in ``free_names`` order; not multiplied by
        χ²/ndf. ``None`` if it could not be computed.
    converged : bool
        Whether the solver met its convergence criteria.
    nfev : int
        Number of model evaluations.
    message : str
        The solver's own status message.
    method : str
        Human-readable solver description for the report.
    raw : object
        The backend's native result object.
    log : tuple of str
        Notes about what the backend did, for ``FitResult.solver_log``.
    """

    values: dict[str, float]
    cov_unscaled: FloatArray | None
    converged: bool
    nfev: int
    message: str
    method: str
    raw: object = None
    log: tuple[str, ...] = field(default_factory=tuple)
    robust_weights: FloatArray | None = None
    """Weights ρ′(z) of a robust fit at the solution (1 = normal point, ≪ 1 = outlier)."""
    chi2: float | None = None
    """χ² of the backend's own objective if it is not Σ((y − f(x))/yerr)², e.g. for ODR."""


class Backend(Protocol):
    """A solver that turns a :class:`FitProblem` into a :class:`BackendResult`."""

    def solve(self, problem: FitProblem) -> BackendResult:
        """Solve the problem."""
        ...
