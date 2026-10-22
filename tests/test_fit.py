"""fitwise.fit(): options, conventions, warnings and agreement with scipy."""

import functools
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import curve_fit

import fitwise as fw
from fitwise import warnings as fww
from fitwise.models import constant, exponential, gaussian, linear

TIGHT = {"xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15}


def gauss_lin(x, amp, cen, sigma, slope, intercept):
    return gaussian(x, amp, cen, sigma) + slope * x + intercept


def assert_same_solution(result, popt):
    """Equal up to a negligible fraction of the standard error (solvers stop at a tolerance)."""
    stderr = np.array([p.stderr for p in result.params.values()])
    np.testing.assert_array_less(np.abs(result.params.as_array() - popt), 1e-5 * stderr)


class TestAgreementWithCurveFit:
    def test_with_yerr_uses_absolute_sigma(self, peak_data):
        x, y, dy = peak_data
        result = fw.fit(gaussian + linear, x, y, yerr=dy)
        popt, pcov = curve_fit(
            gauss_lin,
            x,
            y,
            p0=list(result.initial_guess.values()),
            sigma=dy,
            absolute_sigma=True,
            **TIGHT,
        )
        assert_same_solution(result, popt)
        np.testing.assert_allclose(
            result.covariance, pcov, rtol=1e-4
        )  # finite-difference Jacobians
        assert result.scale_covar is False

    def test_without_yerr_scales_covariance(self, peak_data):
        x, y, _ = peak_data
        result = fw.fit(gaussian + linear, x, y)
        popt, pcov = curve_fit(gauss_lin, x, y, p0=list(result.initial_guess.values()), **TIGHT)
        assert_same_solution(result, popt)
        np.testing.assert_allclose(result.covariance, pcov, rtol=1e-4)
        assert result.scale_covar is True

    def test_scale_covar_override(self, peak_data):
        x, y, dy = peak_data
        absolute = fw.fit(gaussian + linear, x, y, yerr=dy)
        scaled = fw.fit(gaussian + linear, x, y, yerr=dy, scale_covar=True)
        np.testing.assert_allclose(
            scaled.covariance, absolute.covariance * absolute.redchi, rtol=1e-10
        )
        assert "scaled by χ²/ndf" in str(scaled)
        with pytest.warns(fww.LargeUncertaintyWarning):  # unit weights: σ = 1 per point
            unscaled = fw.fit(gaussian + linear, x, y, scale_covar=False)
        assert "covariance not scaled" in str(unscaled)

    def test_lm_method_matches(self, peak_data):
        x, y, dy = peak_data
        trf = fw.fit(gaussian + linear, x, y, yerr=dy)
        lm = fw.fit(gaussian + linear, x, y, yerr=dy, method="lm")
        assert "Levenberg" in lm.method
        assert lm.method_reason == "requested"
        np.testing.assert_allclose(lm.params.as_array(), trf.params.as_array(), rtol=1e-7)


class TestStatistics:
    def test_values(self, peak_data):
        x, y, dy = peak_data
        r = fw.fit(gaussian + linear, x, y, yerr=dy)
        resid = y - r.eval()
        assert r.chi2 == pytest.approx(np.sum((resid / dy) ** 2))
        assert r.rss == pytest.approx(resid @ resid)
        assert r.ndf == 128 - 5
        assert r.redchi == pytest.approx(r.chi2 / 123)
        assert 0.01 < r.pvalue < 0.99
        assert r.residual_sd == pytest.approx(math.sqrt(r.rss / 123))
        assert r.r2 > 0.999
        assert r.aic == pytest.approx(128 * math.log(r.chi2 / 128) + 10)
        assert r.bic == pytest.approx(128 * math.log(r.chi2 / 128) + 5 * math.log(128))
        np.testing.assert_allclose(r.normalized_residuals, resid / dy)
        assert r.ok
        assert r.converged
        assert not r.warnings

    def test_parameters_are_close_to_truth(self, peak_data):
        x, y, dy = peak_data
        p = fw.fit(gaussian + linear, x, y, yerr=dy).params
        for name, truth in {
            "amp": 2.33,
            "cen": 0.21,
            "sigma": 1.51,
            "slope": -0.003,
            "intercept": 0.1,
        }.items():
            assert abs(p[name].value - truth) < 4 * p[name].stderr

    def test_correlation(self, peak_data):
        x, y, dy = peak_data
        r = fw.fit(gaussian + linear, x, y, yerr=dy)
        corr = r.correlation
        np.testing.assert_allclose(np.diag(corr), 1)
        strongest = next(iter(r.correlations(0.5)))
        assert set(strongest) <= set(r.free_names)
        assert all(abs(v) > 0.5 for v in r.correlations(0.5).values())

    def test_exact_data_without_yerr(self):
        x = np.linspace(0, 1, 5)
        r = fw.fit(linear, x, 2 * x + 1)
        assert r.params["slope"].value == pytest.approx(2)
        assert r.chi2 < 1e-20
        assert r.pvalue is None

    def test_zero_dof(self):
        x = np.array([0.0, 1.0])
        with pytest.warns(fww.SingularCovarianceWarning):
            r = fw.fit(linear, x, np.array([1.0, 3.0]))
        assert r.ndf == 0
        assert r.redchi is None
        assert r.residual_sd is None
        assert not r.ok
        assert "0 dof" in str(r)


class TestStartingValues:
    def test_precedence_p0_default_guess_fallback(self):
        m = fw.Model(
            lambda x, amp, tau=2.0, k=1.0: amp * np.exp(-x / tau) + 0 * k,
            guess=lambda x, y: {"amp": 3.0},
        )
        x = np.linspace(0, 5, 30)
        y = 3 * np.exp(-x / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(m, x, y, p0={"k": 1.0})
        assert r.guess_strategy == {"amp": "model heuristic", "tau": "signature default", "k": "p0"}

    def test_fallback_and_bounds(self):
        m = fw.Model(lambda x, a, b: a * np.exp(-b * x))
        x = np.linspace(0, 3, 20)
        r = fw.fit(m, x, 1.5 * np.exp(-0.7 * x), bounds={"a": (0, 4), "b": (0, None)})
        assert r.initial_guess == {"a": 2.0, "b": 1.0}
        assert r.guess_strategy == {"a": "fallback", "b": "fallback"}
        assert r.params["a"].value == pytest.approx(1.5)
        assert "(0, 4)" in str(r)

    def test_fallback_upper_bound_only(self):
        m = fw.Model(lambda x, a: a * x)
        x = np.linspace(0, 1, 10)
        r = fw.fit(m, x, -3 * x, bounds={"a": (None, 0)})
        assert r.initial_guess["a"] == -1.0

    def test_guess_clipped_into_bounds(self):
        x = np.linspace(0, 5, 40)
        with pytest.warns(fww.ParameterAtBoundWarning):
            r = fw.fit(exponential, x, 3 * np.exp(-x / 2), bounds={"tau": (3, 10)})
        assert r.guess_strategy["tau"] == "log-linear fit (clipped to bounds)"
        assert r.params["tau"].value == pytest.approx(3)

    def test_p0_outside_bounds(self):
        with pytest.raises(ValueError, match="outside its bounds"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], p0={"slope": 5}, bounds={"slope": (0, 1)})


class TestFixedParameters:
    def test_fixed_parameter_is_not_varied(self, peak_data):
        x, y, dy = peak_data
        r = fw.fit(gaussian + linear, x, y, yerr=dy, fixed={"slope": 0.0})
        assert r.params["slope"].value == 0.0
        assert r.params["slope"].fixed
        assert r.params["slope"].stderr is None
        assert r.free_names == ("amp", "cen", "sigma", "intercept")
        assert r.ndf == 128 - 4
        assert r.guess_strategy["slope"] == "fixed"
        assert "fixed" in str(r)

    def test_all_fixed(self):
        with pytest.raises(ValueError, match="nothing to fit"):
            fw.fit(linear, [0, 1], [0, 1], fixed={"slope": 1, "intercept": 0})

    @pytest.mark.parametrize("other", [{"p0": {"slope": 1}}, {"bounds": {"slope": (0, 1)}}])
    def test_fixed_and_other_option(self, other):
        with pytest.raises(ValueError, match="is fixed"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], fixed={"slope": 1}, **other)

    def test_fixed_must_be_finite(self):
        with pytest.raises(ValueError, match="finite"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], fixed={"slope": np.inf})


class TestInputValidation:
    def test_unknown_name_suggestion(self):
        with pytest.raises(ValueError, match="Did you mean 'sigma'"):
            fw.fit(gaussian, [0, 1, 2], [0, 1, 0], p0={"sigam": 1})

    def test_non_numeric_p0(self):
        with pytest.raises(TypeError, match="must be a number"):
            fw.fit(gaussian, [0, 1, 2], [0, 1, 0], p0={"sigma": "wide"})

    @pytest.mark.parametrize(
        "bounds", [{"amp": (2, 1)}, {"amp": 3}, {"amp": (0, np.nan)}, {"amp": ("a", 1)}]
    )
    def test_bad_bounds(self, bounds):
        with pytest.raises((ValueError, TypeError)):
            fw.fit(gaussian, [0, 1, 2], [0, 1, 0], bounds=bounds)

    def test_shapes(self):
        with pytest.raises(ValueError, match="same length"):
            fw.fit(linear, [0, 1, 2], [0, 1])
        with pytest.raises(ValueError, match="one-dimensional"):
            fw.fit(linear, [0, 1], [[0, 1], [1, 2]])

    def test_nan_policy(self):
        x = np.array([0.0, 1.0, 2.0, np.nan, 4.0])
        y = 2 * np.nan_to_num(x) + 1
        with pytest.raises(ValueError, match="nan_policy='omit'"):
            fw.fit(linear, x, y)
        r = fw.fit(linear, x, y, nan_policy="omit")
        assert r.data.n == 4
        assert r.data.n_omitted == 1
        assert "1 non-finite omitted" in str(r)
        with pytest.raises(ValueError, match="nan_policy"):
            fw.fit(linear, x, y, nan_policy="drop")  # type: ignore[arg-type]

    def test_yerr_must_be_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], yerr=[1, 0, 1])

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="cannot determine"):
            fw.fit(gaussian, [0, 1], [0, 1])

    def test_model_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            fw.fit(lambda x, a: np.zeros(2) + a, np.arange(5.0), np.arange(5.0))

    def test_scalar_model_output_is_broadcast(self):
        r = fw.fit(lambda x, a: a, np.arange(5.0), 2.0 + np.array([0.1, -0.1, 0, 0.1, -0.1]))
        assert r.params["a"].value == pytest.approx(2.0)

    def test_non_finite_start(self):
        with pytest.raises(ValueError, match="not finite at the starting values"):
            fw.fit(lambda x, a: np.log(a - 2) * x, np.arange(1, 5.0), np.arange(1, 5.0))

    def test_planned_features(self):
        with pytest.raises(NotImplementedError, match=r"0\.3"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], xerr=0.1)
        with pytest.raises(NotImplementedError, match=r"0\.2"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], loss="huber")
        with pytest.raises(NotImplementedError, match=r"0\.2"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], method="multistart")
        with pytest.raises(ValueError, match="unknown method"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], method="newton")  # type: ignore[arg-type]

    def test_lm_with_bounds(self):
        with pytest.raises(ValueError, match="does not support bounds"):
            fw.fit(linear, [0, 1, 2], [0, 1, 2], method="lm", bounds={"slope": (0, 2)})


def test_multidimensional_x():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (200, 2))

    def plane(p, a, b, c):
        return a * p[:, 0] + b * p[:, 1] + c

    z = plane(xy, 1.5, -0.5, 2.0) + rng.normal(0, 0.01, 200)
    r = fw.fit(plane, xy, z, p0={"a": 1, "b": 1, "c": 1})
    np.testing.assert_allclose(r.params.as_array(), [1.5, -0.5, 2.0], atol=0.01)


class TestWarnings:
    def test_not_converged(self, monkeypatch):
        from fitwise.backends import LeastSquaresBackend

        budget = functools.partial(LeastSquaresBackend, max_nfev=2, max_restarts=0)
        monkeypatch.setattr("fitwise.api.LeastSquaresBackend", budget)
        x = np.linspace(0, 5, 30)
        with pytest.warns(fww.NotConvergedWarning):
            r = fw.fit(exponential, x, 3 * np.exp(-x / 2), p0={"amp": 1.0, "tau": 1.0})
        assert not r.ok
        assert "NO (" in str(r)

    def test_redundant_parameters(self):
        x = np.linspace(0, 1, 20)
        with pytest.warns(fww.SingularCovarianceWarning):
            r = fw.fit(lambda x, a, b: (a + b) * x, x, 2 * x + 0.01 * np.sin(9 * x))
        assert not r.ok
        assert all(p.stderr is None for p in r.params.values())
        assert r.correlation is None
        assert r.correlations() == {}

    def test_poor_chi_square_both_directions(self, peak_data):
        x, y, dy = peak_data
        with pytest.warns(fww.PoorChiSquareWarning, match="too large"):
            fw.fit(gaussian + linear, x, y, yerr=dy / 10)
        with pytest.warns(fww.FitWarning) as record:
            fw.fit(gaussian + linear, x, y, yerr=dy * 10)
        poor = [w.message for w in record if isinstance(w.message, fww.PoorChiSquareWarning)]
        assert len(poor) == 1
        assert "too small" in str(poor[0])

    def test_strong_correlation_and_large_uncertainty(self):
        x = np.linspace(1000, 1001, 30)
        y = 1 + 0.01 * np.sin(20 * x)  # slope ≈ 0, far below its uncertainty
        with pytest.warns(fww.FitWarning) as record:
            r = fw.fit(linear, x, y, yerr=0.5)
        kinds = {type(w.message) for w in record}
        assert {fww.StrongCorrelationWarning, fww.LargeUncertaintyWarning} <= kinds
        assert "Warnings:   !" in str(r)

    def test_warning_points_to_caller(self, peak_data):
        x, y, dy = peak_data
        with pytest.warns(fww.PoorChiSquareWarning) as record:
            fw.fit(gaussian + linear, x, y, yerr=dy / 10)
        assert record[0].filename == __file__


def test_constant_plus_exponential_report_mentions_baseline():
    x = np.linspace(0, 10, 60)
    y = 5 * np.exp(-x / 2) + 1 + 0.01 * np.sin(5 * x)
    r = fw.fit(exponential + constant, x, y)
    assert r.guess_strategy["c"] == "edge baseline"
    assert r.params["tau"].value == pytest.approx(2, rel=0.01)
