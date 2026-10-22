"""NIST StRD nonlinear regression: certified values from both official starting points."""

import warnings

import numpy as np
import pytest
from nist_strd import NAMES, load

import fitwise as fw

pytestmark = pytest.mark.nist


def log_relative_error(value, certified):
    """Number of correct significant digits (NIST's LRE), capped at 15."""
    with np.errstate(divide="ignore"):
        lre = -np.log10(np.abs(value - certified) / np.abs(certified))
    return np.minimum(lre, 15)


@pytest.mark.parametrize("start", ["start1", "start2"])
@pytest.mark.parametrize("name", NAMES)
def test_certified_values(name, start):
    ds = load(name)
    p0 = dict(zip(ds.param_names, getattr(ds, start), strict=True))
    with warnings.catch_warnings():
        # strong correlations etc. are real properties of several NIST problems
        warnings.simplefilter("ignore", fw.FitWarning)
        result = fw.fit(fw.Model(ds.model, name=name), ds.x, ds.y, p0=p0)

    assert result.ok
    assert log_relative_error(result.params.as_array(), ds.certified_values).min() >= 6
    if ds.certified_rss < 1e-20:
        # Lanczos1: noise-free data, residuals are floating-point rounding only, so RSS
        # and standard errors are not reproducible beyond a few digits.
        return
    stderr = np.array([p.stderr for p in result.params.values()])
    assert log_relative_error(stderr, ds.certified_stderr).min() >= 4
    assert result.rss == pytest.approx(ds.certified_rss, rel=1e-8)
    assert result.residual_sd == pytest.approx(ds.certified_residual_sd, rel=1e-8)
