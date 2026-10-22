"""Linear models: detection, direct solution, bounds, NIST StRD linear suite."""

import warnings

import numpy as np
import pytest
from nist_linear import NAMES, load
from scipy.optimize import curve_fit

import fitwise as fw
from fitwise import warnings as fww
from fitwise._validate import validate_data
from fitwise.backends.base import FitProblem
from fitwise.backends.linear import is_linear
from fitwise.models import constant, gaussian, linear, polynomial


def problem_for(func, x, init):
    model = fw.Model(func)
    data = validate_data(x, np.zeros(len(x)), None, "raise")
    bounds = {n: (-np.inf, np.inf) for n in init}
    return FitProblem(model, data, init, {}, bounds)


class TestDetection:
    x = np.linspace(-2, 3, 20)

    @pytest.mark.parametrize(
        "func",
        [
            lambda x, a, b: a * x + b,
            lambda x, a, b: a * np.sin(x) + b * x**2 + 3.0,  # affine: constant term allowed
            lambda x, a, b: (a - 2 * b) * np.exp(x),
        ],
    )
    def test_linear_functions(self, func):
        assert is_linear(problem_for(func, self.x, {"a": 1.0, "b": 1.0}))

    @pytest.mark.parametrize(
        "func",
        [
            lambda x, a, b: a * np.exp(-b * x),
            lambda x, a, b: a * b * x,
            lambda x, a, b: a * x + b**2,
            lambda x, a, b: np.log(a) * x + b,  # non-finite for negative test points
        ],
    )
    def test_nonlinear_functions(self, func):
        assert not is_linear(problem_for(func, self.x, {"a": 1.0, "b": 1.0}))

    def test_declared_linearity_is_trusted(self):
        x = self.x
        data = validate_data(x, x, None, "raise")
        free = {"amp": 1.0, "cen": 0.0, "sigma": 1.0}
        problem = FitProblem(gaussian, data, free, {}, dict.fromkeys(free, (-np.inf, np.inf)))
        assert not is_linear(problem)
        poly = polynomial(2)
        problem = FitProblem(
            poly,
            data,
            {"c0": 0, "c1": 0, "c2": 0},
            {},
            dict.fromkeys(poly.param_names, (-np.inf, np.inf)),
        )
        assert is_linear(problem)


class TestDirectSolution:
    def test_report_and_strategies(self, rng):
        x = np.linspace(0, 10, 30)
        r = fw.fit(lambda x, a, b: a * x + b, x, 2 * x + 1 + rng.normal(0, 0.1, x.size), yerr=0.1)
        assert r.method == "Linear least squares (direct)"
        assert r.method_reason == "auto: linear in the parameters"
        assert r.guess_strategy == {
            "a": "not needed (linear model)",
            "b": "not needed (linear model)",
        }
        assert "a, b: not needed (linear model)" in str(r)
        assert r.nfev == 3
        assert r.ok

    def test_matches_curve_fit(self, rng):
        x = np.linspace(-1, 2, 40)
        dy = 0.05 + 0.05 * rng.uniform(size=x.size)
        y = 0.3 - 1.2 * x + 0.8 * x**2 + rng.normal(0, dy)
        r = fw.fit(polynomial(2), x, y, yerr=dy)

        def quad(x, c0, c1, c2):
            return c0 + c1 * x + c2 * x**2

        popt, pcov = curve_fit(quad, x, y, sigma=dy, absolute_sigma=True)
        np.testing.assert_allclose(r.params.as_array(), popt, rtol=1e-7)
        np.testing.assert_allclose(r.covariance, pcov, rtol=1e-6)

    def test_matches_polyfit_without_yerr(self, rng):
        x = np.linspace(0, 5, 25)
        y = 1 + 0.5 * x + 0.3 * x**2 - 0.1 * x**3 + rng.normal(0, 0.2, x.size)
        r = fw.fit(polynomial(3), x, y)
        coeffs, cov = np.polyfit(x, y, 3, cov=True)
        np.testing.assert_allclose(r.params.as_array(), coeffs[::-1], rtol=1e-9)
        np.testing.assert_allclose(r.covariance, cov[::-1, ::-1], rtol=1e-9)

    def test_fixed_parameter(self):
        x = np.linspace(0, 1, 10)
        r = fw.fit(linear, x, 3 * x + 0.2, fixed={"intercept": 0.0})
        assert r.free_names == ("slope",)
        assert r.method_reason == "auto: linear in the parameters"
        assert r.params["intercept"].value == 0.0

    def test_composite_of_linear_models(self):
        x = np.linspace(0, 1, 10)
        r = fw.fit(constant + fw.Model(lambda x, k: k * x**2), x, 1 + 2 * x**2)
        assert r.method.startswith("Linear")
        np.testing.assert_allclose(r.params.as_array(), [1, 2], atol=1e-12)

    def test_bounds_use_lsq_linear(self):
        x = np.linspace(0, 1, 20)
        with pytest.warns(fww.ParameterAtBoundWarning):
            r = fw.fit(linear, x, 2 * x + 1 + 0.01 * np.sin(30 * x), bounds={"slope": (0, 1.5)})
        assert r.method == "Bounded linear least squares (scipy lsq_linear)"
        assert r.params["slope"].value == pytest.approx(1.5)

    def test_bounds_inactive(self):
        x = np.linspace(0, 1, 20)
        r = fw.fit(linear, x, 2 * x + 1 + 0.01 * np.sin(30 * x), bounds={"slope": (0, None)})
        free = fw.fit(linear, x, 2 * x + 1 + 0.01 * np.sin(30 * x))
        np.testing.assert_allclose(r.params.as_array(), free.params.as_array(), rtol=1e-9)

    def test_explicit_methods(self):
        x = np.linspace(0, 1, 10)
        y = 2 * x + 1 + 0.01 * np.cos(9 * x)
        assert fw.fit(linear, x, y, method="linear").method_reason == "requested"
        trf = fw.fit(linear, x, y, method="trf")
        assert trf.method.startswith("Trust Region")
        with pytest.raises(ValueError, match="linear in its parameters"):
            fw.fit(gaussian, x, y, method="linear")

    def test_multidimensional_x(self, rng):
        xy = rng.uniform(-1, 1, (50, 2))
        z = 1.5 * xy[:, 0] - 0.5 * xy[:, 1] + 2 + rng.normal(0, 0.01, 50)
        r = fw.fit(lambda p, a, b, c: a * p[:, 0] + b * p[:, 1] + c, xy, z)
        assert r.method.startswith("Linear")
        np.testing.assert_allclose(r.params.as_array(), [1.5, -0.5, 2], atol=0.01)


@pytest.mark.nist
@pytest.mark.parametrize("name", NAMES)
def test_nist_linear(name):
    """Certified values: ≥ 7 digits, or deviation far below the certified standard error."""
    ds = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fw.FitWarning)
        result = fw.fit(fw.Model(ds.model, param_names=ds.param_names), ds.x, ds.y)

    assert result.method_reason == "auto: linear in the parameters"
    values = result.params.as_array()
    deviation = np.abs(values - ds.certified_values)
    with np.errstate(divide="ignore"):
        digits = -np.log10(deviation / np.abs(ds.certified_values))
    assert np.all((digits >= 7) | (deviation <= 1e-6 * ds.certified_stderr))

    stderr = np.array([p.stderr for p in result.params.values()])
    np.testing.assert_allclose(stderr, ds.certified_stderr, rtol=1e-7, atol=1e-9)
    assert result.residual_sd == pytest.approx(ds.certified_residual_sd, rel=1e-8, abs=1e-9)
