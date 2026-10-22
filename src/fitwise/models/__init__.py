"""Built-in models with documented parameters and starting-value heuristics.

Models can be combined: ``gaussian + linear`` fits a peak on a sloped background.
"""

from fitwise.models.basic import constant, linear, polynomial
from fitwise.models.curves import damped_sine, exponential, powerlaw, sigmoid
from fitwise.models.peaks import gaussian, lorentzian, voigt

__all__ = [
    "constant",
    "damped_sine",
    "exponential",
    "gaussian",
    "linear",
    "lorentzian",
    "polynomial",
    "powerlaw",
    "sigmoid",
    "voigt",
]
