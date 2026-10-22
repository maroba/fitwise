"""Robust fits with outliers."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fitwise as fw
from fitwise.backends.least_squares import LeastSquaresBackend, rho_prime
from fitwise.models import linear


@pytest.fixture
def line_with_outliers(rng):
    x = np.linspace(0, 10, 60)
    y = 2 * x + 1 + rng.normal(0, 0.3, x.size)
    y[[7, 23, 41]] += [6.0, -5.0, 8.0]
    return x, y


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy", "arctan"])
def test_resists_outliers(line_with_outliers, loss):
    x, y = line_with_outliers
    with pytest.warns(fw.warnings.PoorChiSquareWarning):
        ordinary = fw.fit(linear, x, y, yerr=0.3)
    robust = fw.fit(linear, x, y, yerr=0.3, loss=loss)
    assert not robust.warnings
    for name, truth in (("slope", 2.0), ("intercept", 1.0)):
        p = robust.params[name]
        assert abs(p.value - truth) < 3 * p.stderr
    assert abs(robust.params["intercept"].value - 1) < abs(ordinary.params["intercept"].value - 1)
    assert set(np.flatnonzero(robust.downweighted)) >= {7, 23, 41}
    assert robust.loss == loss


def test_weights_match_scipy_losses():
    from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES

    z = np.array([0.0, 0.3, 1.0, 4.0, 100.0])
    for loss in ("soft_l1", "huber", "cauchy", "arctan"):
        rho = np.empty((3, z.size))
        IMPLEMENTED_LOSSES[loss](z, rho, cost_only=False)
        np.testing.assert_allclose(rho_prime(loss, z), rho[1])
    np.testing.assert_array_equal(rho_prime("linear", z), np.ones_like(z))
    with pytest.raises(ValueError, match="unknown loss"):
        rho_prime("tukey", z)


def test_report(line_with_outliers):
    x, y = line_with_outliers
    r = fw.fit(linear, x, y, yerr=0.3, loss="huber")
    text = str(r)
    assert "Robust fit (huber loss, scale 1.5)" in text
    assert "auto: linear in the parameters, robust (huber)" in text
    assert "χ²/ndf (robust) = " in text
    assert "Robust:     huber loss; " in text
    assert r.pvalue is None
    assert any(
        "stage 2: huber loss, scale 1.5 (default, in units of yerr)" in s for s in r.solver_log
    )


def test_without_yerr_uses_robust_scatter(line_with_outliers):
    x, y = line_with_outliers
    r = fw.fit(linear, x, y, loss="cauchy")
    assert "uncertainties from robust residual scatter" in str(r)
    assert "σ_res (robust) = " in str(r)
    assert np.sqrt(r.redchi) == pytest.approx(0.3, rel=0.3)
    assert any("1.5 × MAD" in s for s in r.solver_log)


def test_given_scale(line_with_outliers):
    x, y = line_with_outliers
    r = fw.fit(linear, x, y, yerr=0.3, loss="huber", loss_scale=3.0)
    assert "scale 3" in r.method
    assert any("(given)" in s for s in r.solver_log)


def test_clean_data_matches_ordinary_fit(rng):
    x = np.linspace(0, 10, 60)
    y = 2 * x + 1 + rng.normal(0, 0.3, x.size)
    ordinary = fw.fit(linear, x, y, yerr=0.3)
    robust = fw.fit(linear, x, y, yerr=0.3, loss="soft_l1", loss_scale=100.0)
    np.testing.assert_allclose(robust.params.as_array(), ordinary.params.as_array(), rtol=1e-4)
    np.testing.assert_allclose(robust.covariance, ordinary.covariance, rtol=1e-3)


def test_nonlinear_model_with_multistart(rng):
    x = np.linspace(0, 10, 60)
    y = 3 * np.exp(-x / 2.5) + 0.4 + rng.normal(0, 0.02, x.size)
    y[::15] += 0.5
    r = fw.fit("a*exp(-x/tau) + c", x, y, loss="cauchy")
    assert r.method_reason == "auto: nonlinear, no starting values, robust (cauchy)"
    np.testing.assert_allclose(r.params.as_array(), [3, 2.5, 0.4], rtol=0.03)


@pytest.mark.slow
@pytest.mark.parametrize("loss", ["huber", "cauchy"])
@pytest.mark.parametrize("with_yerr", [True, False])
def test_uncertainties_are_calibrated(loss, with_yerr):
    """Pulls (fit − truth)/σ over 150 data sets with 5 % outliers have a spread of about 1."""
    x = np.linspace(0, 10, 60)
    pulls = []
    for seed in range(150):
        rng = np.random.default_rng(seed)
        y = 2 * x + 1 + rng.normal(0, 0.3, x.size)
        idx = rng.choice(60, 3, replace=False)
        y[idx] += rng.choice([-1, 1], 3) * rng.uniform(3, 10, 3)
        r = fw.fit(linear, x, y, yerr=0.3 if with_yerr else None, loss=loss)
        pulls.append((r.params["slope"].value - 2) / r.params["slope"].stderr)
    assert abs(np.mean(pulls)) < 0.2
    assert 0.8 < np.std(pulls) < 1.25


def test_invalid_options():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="unknown loss"):
        fw.fit(linear, x, x, loss="tukey")
    with pytest.raises(ValueError, match="loss_scale must be positive"):
        fw.fit(linear, x, x, loss="huber", loss_scale=0)
    with pytest.raises(ValueError, match="does not support robust losses"):
        fw.fit(linear, x, x, loss="huber", method="lm")
    from fitwise._validate import validate_data
    from fitwise.backends.base import FitProblem

    data = validate_data(x, x, None, "raise")
    unbounded = (-np.inf, np.inf)
    problem = FitProblem(
        linear,
        data,
        {"slope": 1, "intercept": 0},
        {},
        dict.fromkeys(("slope", "intercept"), unbounded),
    )
    with pytest.raises(ValueError, match="does not support robust losses"):
        LeastSquaresBackend("lm", loss="huber").solve(problem)


def test_plot_and_export(line_with_outliers):
    x, y = line_with_outliers
    r = fw.fit(linear, x, y, yerr=0.3, loss="huber")
    fig, axes = r.plot()
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert "downweighted (huber)" in labels
    plt.close(fig)
    d = r.to_dict(include_data=True)
    assert json.loads(json.dumps(d)) == d
    assert d["robust"]["loss"] == "huber"
    assert d["robust"]["n_downweighted"] == int(r.downweighted.sum())
    assert len(d["robust"]["weights"]) == 60
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fw.FitWarning)
        ordinary = fw.fit(linear, x, y)
    assert ordinary.to_dict()["robust"] is None
    assert not ordinary.downweighted.any()
