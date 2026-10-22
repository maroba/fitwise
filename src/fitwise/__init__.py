"""fitwise: curve fitting made easy.

One call, one fit, one report you can trust.

>>> import fitwise as fw
>>> result = fw.fit(fw.models.gaussian, x, y, yerr=dy)  # doctest: +SKIP
>>> print(result)  # doctest: +SKIP
"""

from fitwise import models, warnings
from fitwise.api import fit
from fitwise.expression import ExpressionModel
from fitwise.model import CompositeModel, Model
from fitwise.parameters import Parameter, Parameters
from fitwise.result import FitResult
from fitwise.warnings import FitWarning

__version__ = "0.2.0.dev0"

__all__ = [
    "CompositeModel",
    "ExpressionModel",
    "FitResult",
    "FitWarning",
    "Model",
    "Parameter",
    "Parameters",
    "__version__",
    "fit",
    "models",
    "warnings",
]
