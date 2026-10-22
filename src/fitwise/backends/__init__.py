"""Solver backends behind :func:`fitwise.fit`."""

from fitwise.backends.base import Backend, BackendResult, FitProblem
from fitwise.backends.least_squares import LeastSquaresBackend
from fitwise.backends.linear import LinearBackend

__all__ = ["Backend", "BackendResult", "FitProblem", "LeastSquaresBackend", "LinearBackend"]
