"""Sanity checks for the vendored NIST StRD datasets and their model functions.

These tests do not fit anything. They guarantee that the reference data that later
validates fitwise is itself parsed and modelled correctly.
"""

import numpy as np
import pytest
from nist_strd import NAMES, load, load_all


@pytest.mark.parametrize("name", NAMES)
def test_certified_values_reproduce_certified_rss(name):
    ds = load(name)
    assert ds.rss(ds.certified_values) == pytest.approx(ds.certified_rss, rel=1e-8)


# Rat43.dat states 9 degrees of freedom, but 15 observations - 4 parameters = 11, and its
# certified residual standard deviation is computed with 11. The file's DoF line is a typo.
KNOWN_DOF_ERRATA = {"Rat43"}


@pytest.mark.parametrize("name", NAMES)
def test_metadata_is_consistent(name):
    ds = load(name)
    k = len(ds.param_names)
    assert ds.param_names == tuple(f"b{i}" for i in range(1, k + 1))
    assert ds.start1.shape == ds.start2.shape == ds.certified_values.shape == (k,)
    assert (ds.dof_stated == ds.dof) != (name in KNOWN_DOF_ERRATA)
    assert np.sqrt(ds.certified_rss / ds.dof) == pytest.approx(ds.certified_residual_sd, rel=1e-8)
    assert np.all(np.isfinite(ds.x))
    assert np.all(np.isfinite(ds.y))


def test_difficulty_counts_match_nist():
    assert len(load_all("lower")) == 8
    assert len(load_all("average")) == 11
    assert len(load_all("higher")) == 8


def test_nelson_has_two_predictors_and_log_response():
    ds = load("Nelson")
    assert ds.x.shape == (ds.n_obs, 2)
    assert np.all(ds.y < 3)  # log of dielectric breakdown strength (raw values up to ~16)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError, match="unknown NIST dataset"):
        load("Nope")
