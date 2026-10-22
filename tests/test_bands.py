"""Confidence and prediction bands, correlated ufloats, calibration of uncertainties."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

import fitwise as fw
from fitwise.models import constant, exponential, gaussian, linear


@pytest.fixture
def decay_fit(rng):
    x = np.linspace(0, 10, 40)
    y = 3 * np.exp(-x / 2.5) + 0.4 + rng.normal(0, 0.05, x.size)
    return fw.fit(exponential + constant, x, y, yerr=0.05)


class TestBands:
    def test_linear_model_matches_closed_form(self, rng):
        x = np.linspace(0, 5, 30)
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size), yerr=0.2)
        xn = np.array([-1.0, 2.5, 8.0])
        design = np.column_stack([xn, np.ones_like(xn)])
        sd = np.sqrt(np.einsum("ij,jk,ik->i", design, r.covariance, design))
        band = r.confidence_band(xn, level=0.6827)
        np.testing.assert_allclose(
            (band.upper - band.lower) / 2, sd * stats.norm.ppf(0.84135), rtol=1e-6
        )
        np.testing.assert_allclose(band.center, r.eval(xn))
        assert band.kind == "confidence"
        assert band.level == 0.6827

    def test_student_t_without_yerr(self, rng):
        x = np.linspace(0, 5, 8)
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size))
        b95 = r.confidence_band([1.0])
        design = np.array([[1.0, 1.0]])
        sd = np.sqrt(design @ r.covariance @ design.T)[0, 0]
        assert (b95.upper - b95.center)[0] == pytest.approx(stats.t.ppf(0.975, 6) * sd, rel=1e-6)

    def test_prediction_band_is_wider(self, decay_fit):
        xn = np.linspace(0, 10, 5)
        conf = decay_fit.confidence_band(xn)
        pred = decay_fit.prediction_band(xn)
        np.testing.assert_array_less(conf.upper - conf.lower, pred.upper - pred.lower)
        half_noise = stats.norm.ppf(0.975) * 0.05
        assert np.all((pred.upper - pred.center) >= half_noise)

    def test_prediction_band_with_varying_yerr(self, rng):
        x = np.linspace(0, 5, 30)
        dy = 0.1 + 0.05 * x
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, dy), yerr=dy)
        with pytest.raises(ValueError, match="yerr_new"):
            r.prediction_band([1.0])
        assert r.prediction_band([1.0, 2.0], yerr_new=[0.1, 0.2]).kind == "prediction"
        fig, _ = r.plot(band="prediction")  # band silently skipped
        plt.close(fig)

    def test_prediction_band_without_yerr(self, rng):
        x = np.linspace(0, 5, 30)
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size))
        band = r.prediction_band([2.0])
        assert (band.upper - band.center)[0] > stats.t.ppf(0.975, 28) * r.residual_sd

    def test_errors(self, decay_fit):
        with pytest.raises(ValueError, match="level"):
            decay_fit.confidence_band([1.0], level=1.5)
        x = np.linspace(0, 1, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            singular = fw.fit(lambda x, a, b: (a + b) * x, x, 2 * x + 0.01 * np.sin(9 * x))
        with pytest.raises(ValueError, match="covariance"):
            singular.confidence_band([0.5])

    def test_tiny_parameters(self, rng):
        """Relative difference steps: a parameter of 1e-12 must not break the Jacobian."""
        v = np.linspace(0.3, 0.7, 30)

        def shockley(v, i_s, n):
            return i_s * (np.exp(v / (n * 0.025852)) - 1)

        i = shockley(v, 5e-12, 1.4) * (1 + rng.normal(0, 0.01, v.size))
        r = fw.fit(shockley, v, i, yerr=0.01 * i)
        band = r.confidence_band(v)
        rel = (band.upper - band.lower) / 2 / band.center
        assert np.all((rel > 1e-4) & (rel < 0.05))

    def test_plot_shows_band(self, decay_fit):
        fig, axes = decay_fit.plot()
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert "95 % confidence band" in labels
        plt.close(fig)
        fig, axes = decay_fit.plot(band="prediction", level=0.68)
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert "68 % prediction band" in labels
        plt.close(fig)
        fig, axes = decay_fit.plot(band=None)
        assert not any("band" in t.get_text() for t in axes[0].get_legend().get_texts())
        plt.close(fig)


class TestUfloat:
    def test_correlated_propagation_matches_delta_method(self, decay_fit):
        u = decay_fit.params.as_ufloat()
        product = u["amp"] * u["tau"]
        cov = decay_fit.covariance
        i, j = decay_fit.free_names.index("amp"), decay_fit.free_names.index("tau")
        a, t = decay_fit.params["amp"].value, decay_fit.params["tau"].value
        grad = np.array([t, a])
        sub = cov[np.ix_([i, j], [i, j])]
        assert product.std_dev == pytest.approx(np.sqrt(grad @ sub @ grad), rel=1e-9)

    def test_fixed_parameters_are_floats(self, rng):
        x = np.linspace(0, 1, 20)
        r = fw.fit(linear, x, 2 * x + rng.normal(0, 0.01, x.size), fixed={"intercept": 0.0})
        u = r.params.as_ufloat()
        assert u["intercept"] == 0.0
        assert u["slope"].std_dev > 0

    def test_errors(self, decay_fit, monkeypatch):
        import builtins

        x = np.linspace(0, 1, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            singular = fw.fit(lambda x, a, b: (a + b) * x, x, 2 * x + 0.01 * np.sin(9 * x))
        with pytest.raises(ValueError, match="covariance"):
            singular.params.as_ufloat()
        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.startswith("uncertainties"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(ImportError, match=r"fitwise\[units\]"):
            decay_fit.params.as_ufloat()


CALIBRATION_CASES = {
    "gaussian, yerr": (
        gaussian,
        np.linspace(-4, 4, 50),
        {"amp": 2.0, "cen": 0.3, "sigma": 0.8},
        True,
    ),
    "gaussian, no yerr": (
        gaussian,
        np.linspace(-4, 4, 50),
        {"amp": 2.0, "cen": 0.3, "sigma": 0.8},
        False,
    ),
    "decay, yerr": (
        exponential + constant,
        np.linspace(0, 10, 40),
        {"amp": 3.0, "tau": 2.5, "c": 0.4},
        True,
    ),
    "line, no yerr": (linear, np.linspace(0, 5, 12), {"slope": 2.0, "intercept": 1.0}, False),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CALIBRATION_CASES))
def test_calibration(case):
    """Monte Carlo: 1σ intervals cover the truth 68 %, 95 % bands cover curve and new points."""
    model, x, truth, with_yerr = CALIBRATION_CASES[case]
    sigma = 0.05
    y_true = model.eval(x, truth)
    x_check = np.array([x[len(x) // 3], x[len(x) // 2]])
    f_check = model.eval(x_check, truth)
    hits, conf_hits, pred_hits, n = 0, 0, 0, 400
    for seed in range(n):
        rng = np.random.default_rng(seed)
        y = y_true + rng.normal(0, sigma, x.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(model, x, y, yerr=sigma if with_yerr else None)
        p = r.params
        hits += sum(abs(p[k].value - v) <= p[k].stderr for k, v in truth.items())
        conf = r.confidence_band(x_check)
        conf_hits += int(np.sum((conf.lower <= f_check) & (f_check <= conf.upper)))
        new = f_check + rng.normal(0, sigma, f_check.size)
        pred = r.prediction_band(x_check)
        pred_hits += int(np.sum((pred.lower <= new) & (new <= pred.upper)))
    coverage = hits / (n * len(truth))
    # ±1 standard error covers 68.3 % if σ is known (yerr), but only P(|t_ndf| ≤ 1) if σ is
    # estimated from the residuals: 65.9 % for the 10 degrees of freedom of the line case
    ndf = len(x) - len(truth)
    expected = stats.norm.cdf(1) * 2 - 1 if with_yerr else stats.t.cdf(1, ndf) * 2 - 1
    assert abs(coverage - expected) <= 0.03, (coverage, expected)
    assert 0.93 <= conf_hits / (2 * n) <= 0.97
    assert 0.93 <= pred_hits / (2 * n) <= 0.97
