"""fitwise: curve fitting made easy.

One call, one fit, one report you can trust.

>>> import fitwise as fw
>>> result = fw.fit(fw.models.gaussian, x, y, yerr=dy)  # doctest: +SKIP
>>> print(result)  # doctest: +SKIP
"""

from fitwise import models, warnings
from fitwise.api import fit
from fitwise.model import CompositeModel, Model
from fitwise.parameters import Parameter, Parameters
from fitwise.result import FitResult
from fitwise.warnings import FitWarning

__version__ = "0.1.1"

__all__ = [
    "CompositeModel",
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
