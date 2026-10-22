"""Orthogonal distance regression: uncertainties in x and y."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fitwise as fw
from fitwise.models import constant, exponential, linear
from fitwise.serialize import data_hash

# Pearson's data with York's weights (York 1966; solution in York et al. 2004, Am. J. Phys.)
X = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
Y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
WX = np.array([1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0])
WY = np.array([1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500.0])
XERR, YERR = 1 / np.sqrt(WX), 1 / np.sqrt(WY)


@pytest.fixture
def york():
    return fw.fit(linear, X, Y, yerr=YERR, xerr=XERR)


class TestReference:
    def test_york_pearson_solution(self, york):
        assert york.params["slope"].value == pytest.approx(-0.480533, abs=2e-6)
        assert york.params["intercept"].value == pytest.approx(5.47991, abs=2e-5)
        assert york.chi2 == pytest.approx(11.8663, rel=1e-4)
        assert york.ndf == 8
        # York's closed-form errors are 0.0577 and 0.2942; ODR's Jacobian-based ones differ
        # by about 0.5 %
        assert york.params["slope"].stderr == pytest.approx(0.0577, rel=0.01)
        assert york.params["intercept"].stderr == pytest.approx(0.2942, rel=0.01)
        assert york.ok
        assert not york.warnings

    def test_matches_scipy_odr(self, rng):
        odr = pytest.importorskip("scipy.odr")
        x_true = np.linspace(0, 8, 30)
        xerr, yerr = 0.1, 0.05
        x = x_true + rng.normal(0, xerr, x_true.size)
        y = 3 * np.exp(-x_true / 2.5) + 0.4 + rng.normal(0, yerr, x_true.size)
        r = fw.fit(exponential + constant, x, y, yerr=yerr, xerr=xerr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = odr.Model(lambda b, x: b[0] * np.exp(-x / b[1]) + b[2])
            data = odr.RealData(x, y, sx=np.full_like(x, xerr), sy=np.full_like(y, yerr))
            out = odr.ODR(data, model, beta0=r.params.as_array()).run()
        np.testing.assert_allclose(r.params.as_array(), out.beta, rtol=1e-5)
        np.testing.assert_allclose(r.covariance, out.cov_beta, rtol=1e-3)

    def test_tiny_xerr_gives_ordinary_fit(self, rng):
        x = np.linspace(0, 5, 30)
        y = 2 * x + 1 + rng.normal(0, 0.2, x.size)
        ordinary = fw.fit(linear, x, y, yerr=0.2)
        odr = fw.fit(linear, x, y, yerr=0.2, xerr=1e-9)
        np.testing.assert_allclose(odr.params.as_array(), ordinary.params.as_array(), rtol=1e-6)
        np.testing.assert_allclose(odr.covariance, ordinary.covariance, rtol=1e-4)
        assert odr.chi2 == pytest.approx(ordinary.chi2, rel=1e-6)


class TestBehaviour:
    def test_report_and_strategies(self, york):
        text = str(york)
        assert "Orthogonal distance regression (odrpack) (auto: xerr given)" in text
        assert "10 points, xerr and yerr given (absolute)" in text
        assert york.guess_strategy["slope"] == "from ordinary fit (linear model)"
        assert any("stage 2: orthogonal distance regression" in s for s in york.solver_log)

    def test_nonlinear_model_without_starting_values(self, rng):
        def decay(t, a, tau):
            return a * np.exp(-t / tau)

        t = np.linspace(0, 10, 40)
        y = decay(t, 3, 2.5) + rng.normal(0, 0.02, t.size)
        r = fw.fit(decay, t + rng.normal(0, 0.05, t.size), y, yerr=0.02, xerr=0.05)
        assert r.guess_strategy["tau"] == "from ordinary fit (multi-start)"
        assert r.params["tau"].value == pytest.approx(2.5, rel=0.05)

    def test_explicit_method(self):
        r = fw.fit(linear, X, Y, yerr=YERR, xerr=XERR, method="odr")
        assert r.method_reason == "requested"

    def test_effective_errors(self, york):
        slope = york.params["slope"].value
        np.testing.assert_allclose(york.effective_yerr, np.hypot(YERR, slope * XERR), rtol=1e-6)
        np.testing.assert_allclose(york.normalized_residuals, york.residuals / york.effective_yerr)

    def test_inactive_bounds(self, york):
        r = fw.fit(linear, X, Y, yerr=YERR, xerr=XERR, bounds={"slope": (-1, 0)})
        np.testing.assert_allclose(r.params.as_array(), york.params.as_array(), rtol=1e-6)
        assert r.ok

    def test_active_bound_explains_the_workaround(self):
        """ODRPACK fails if the solution lies on a bound; fitwise says what to do instead."""
        with pytest.warns(fw.warnings.FitWarning) as record:
            r = fw.fit(linear, X, Y, yerr=YERR, xerr=XERR, bounds={"slope": (-0.4, 0)})
        assert not r.ok
        assert "hold the parameter at the bound value with fixed=" in r.message
        assert any(isinstance(w.message, fw.warnings.NotConvergedWarning) for w in record)
        held = fw.fit(linear, X, Y, yerr=YERR, xerr=XERR, fixed={"slope": -0.4})
        assert held.ok
        odr = pytest.importorskip("scipy.odr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = odr.Model(lambda b, x: b[0] * x + b[1])
            out = odr.ODR(
                odr.RealData(X, Y, sx=XERR, sy=YERR), model, beta0=[-0.4, 5.0], ifixb=[0, 1]
            ).run()
        assert held.params["intercept"].value == pytest.approx(out.beta[1], rel=1e-6)

    def test_redundant_parameters(self):
        with pytest.warns(fw.warnings.FitWarning) as record:
            r = fw.fit(lambda x, a, b: (a + b) * x + 5.5, X, Y, yerr=YERR, xerr=XERR)
        kinds = {type(w.message) for w in record}
        assert fw.warnings.SingularCovarianceWarning in kinds
        assert fw.warnings.NotConvergedWarning not in kinds
        assert r.covariance is None
        assert r.converged  # parameters converged; only the covariance is undetermined

    def test_plot_bands_export(self, york):
        fig, axes = york.plot()
        container = axes[0].containers[0]
        assert container.has_xerr
        assert container.has_yerr
        plt.close(fig)
        band = york.confidence_band([0.0, 4.0])
        assert np.all(band.upper > band.lower)
        d = york.to_dict(include_data=True)
        assert json.loads(json.dumps(d)) == d
        assert d["data"]["has_xerr"] is True
        assert d["data"]["xerr"] == pytest.approx(XERR.tolist())
        assert d["data"]["sha256"] != data_hash(X, Y, YERR)
        assert data_hash(X, Y, YERR) == data_hash(X, Y, YERR, None)


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"xerr": 0.1}, "xerr needs yerr"),
            ({"xerr": 0.1, "yerr": 0.1, "method": "lm"}, "ignores xerr"),
            ({"xerr": 0.1, "yerr": 0.1, "loss": "huber"}, "robust losses"),
            ({"yerr": 0.1, "method": "odr"}, "needs xerr"),
            ({"xerr": 0.0, "yerr": 0.1}, "xerr must be strictly positive"),
        ],
    )
    def test_invalid(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            fw.fit(linear, X, Y, **kwargs)

    def test_two_dimensional_x(self):
        with pytest.raises(ValueError, match="one-dimensional x"):
            fw.fit(lambda p, a: a * p[:, 0], np.ones((5, 2)), np.ones(5), yerr=0.1, xerr=0.1)

    def test_nan_in_xerr(self):
        xerr = XERR.copy()
        xerr[3] = np.nan
        with pytest.raises(ValueError, match="xerr"):
            fw.fit(linear, X, Y, yerr=YERR, xerr=xerr)
        r = fw.fit(linear, X, Y, yerr=YERR, xerr=xerr, nan_policy="omit")
        assert r.data.n == 9

    def test_missing_odrpack(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "odrpack":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        with pytest.raises(ImportError, match=r"fitwise\[odr\]"):
            fw.fit(linear, X, Y, yerr=YERR, xerr=XERR)


@pytest.mark.slow
def test_calibration():
    """Pulls of slope and intercept over 300 data sets with errors in x and y."""
    x_true = np.linspace(0, 10, 25)
    xerr, yerr = 0.3, 0.2
    pulls = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        x = x_true + rng.normal(0, xerr, x_true.size)
        y = 1.5 * x_true - 2 + rng.normal(0, yerr, x_true.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)  # rare χ² fluctuations
            r = fw.fit(linear, x, y, yerr=yerr, xerr=xerr)
        p = r.params
        pulls.append(
            [
                (p["slope"].value - 1.5) / p["slope"].stderr,
                (p["intercept"].value + 2) / p["intercept"].stderr,
            ]
        )
    pulls = np.array(pulls)
    assert np.all(np.abs(pulls.mean(axis=0)) < 0.2)
    assert np.all(np.abs(pulls.std(axis=0) - 1) < 0.12)
