"""Warnings that flag fit results which should not be trusted blindly.

Every warning is stored in :attr:`fitwise.FitResult.warnings`, printed in the report and
emitted through Python's :mod:`warnings` machinery. Silence them with the standard filters,
for example ``warnings.simplefilter("ignore", fitwise.warnings.StrongCorrelationWarning)``.
"""

__all__ = [
    "FitWarning",
    "LargeUncertaintyWarning",
    "NotConvergedWarning",
    "ParameterAtBoundWarning",
    "PoorChiSquareWarning",
    "SingularCovarianceWarning",
    "StrongCorrelationWarning",
]


class FitWarning(UserWarning):
    """Base class of all fitwise warnings."""


class NotConvergedWarning(FitWarning):
    """The solver stopped without meeting its convergence criteria."""


class SingularCovarianceWarning(FitWarning):
    """The covariance matrix could not be estimated; parameters may be unidentifiable."""


class ParameterAtBoundWarning(FitWarning):
    """A parameter ended at one of its bounds; its uncertainty is unreliable."""


class PoorChiSquareWarning(FitWarning):
    """χ²/ndf is implausible for the given ``yerr``: wrong model or wrong uncertainties."""


class StrongCorrelationWarning(FitWarning):
    """Two parameters are almost fully correlated; the model may be over-parametrized."""


class LargeUncertaintyWarning(FitWarning):
    """A parameter's standard error exceeds its absolute value."""
