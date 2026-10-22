"""Built-in models: values, heuristics and convergence without starting values."""

import warnings

import numpy as np
import pytest
from synthetic import CASES

import fitwise as fw
from fitwise.models import _heuristics as h


def test_peak_heights_and_widths():
    x = np.linspace(-10, 10, 20001)
    for model, params, fwhm in [
        (fw.models.gaussian, (2.0, 1.0, 0.5), 2 * np.sqrt(2 * np.log(2)) * 0.5),
        (fw.models.lorentzian, (2.0, 1.0, 0.5), 1.0),
    ]:
        y = model(x, *params)
        assert y.max() == pytest.approx(2.0)
        above = x[y >= 1.0]
        assert above[-1] - above[0] == pytest.approx(fwhm, abs=2e-3)
    assert fw.models.voigt(np.array([1.0]), 3.0, 1.0, 0.4, 0.2)[0] == pytest.approx(3.0)


def test_voigt_limits():
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(
        fw.models.voigt(x, 1, 0, 0.5, 1e-9), fw.models.gaussian(x, 1, 0, 0.5), atol=1e-6
    )


def test_polynomial():
    p = fw.models.polynomial(2)
    assert p.name == "polynomial2"
    np.testing.assert_allclose(p(np.array([2.0]), 1, 2, 3), [17.0])
    assert fw.models.polynomial(0)(np.zeros(3), 4.0).tolist() == [4.0, 4.0, 4.0]
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            fw.models.polynomial(bad)


def test_polynomial_guess_with_too_few_unique_points():
    guess = fw.models.polynomial(3).guess(np.array([1.0, 1.0, 2.0]), np.array([1.0, 1.0, 3.0]))
    assert guess["c2"] == guess["c3"] == 0.0


def test_linear_guess_degenerate_x():
    assert fw.models.linear.guess(np.ones(3), np.array([1.0, 2.0, 3.0])) == {
        "slope": 0.0,
        "intercept": 2.0,
    }


def test_sigmoid_is_overflow_safe():
    with np.errstate(over="raise"):
        y = fw.models.sigmoid(np.array([-1e6, 1e6]), 1.0, 0.0, 1.0)
    np.testing.assert_allclose(y, [0.0, 1.0], atol=1e-300)


class TestHeuristics:
    def test_peak_at_edge(self):
        x = np.linspace(0, 5, 51)
        peak = h.dominant_peak(x, np.exp(-x))
        assert peak.position == 0.0
        assert peak.fwhm == pytest.approx(2 * np.log(2), rel=0.05)

    def test_negative_peak(self):
        x = np.linspace(-3, 3, 61)
        peak = h.dominant_peak(x, -fw.models.gaussian(x, 2, 1, 0.5))
        assert peak.height == pytest.approx(-2)
        assert peak.position == pytest.approx(1)

    def test_flat_data(self):
        x = np.linspace(0, 1, 5)
        assert h.dominant_peak(x, np.ones(5)).fwhm == pytest.approx(0.5)

    def test_loglinear_needs_points(self):
        with pytest.raises(ValueError, match="too few points"):
            h.loglinear(np.array([0.0, 1.0]), np.array([1.0, -1.0]))

    def test_multidimensional_x_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            h.sorted_xy(np.zeros((3, 2)), np.zeros(3))
        assert fw.models.gaussian.guess(np.zeros((3, 2)), np.zeros(3)) == {}

    def test_frequency_needs_range(self):
        with pytest.raises(ValueError, match="span a range"):
            h.dominant_frequency(np.zeros(4), np.arange(4.0))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CASES))
def test_converges_without_starting_values(name):
    """Every built-in model and common composite fits random data without p0."""
    rng = np.random.default_rng(2026)
    failures = []
    for i in range(100):
        case = CASES[name](rng)
        y, dy = case.sample(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            result = fw.fit(case.model, case.x, y, yerr=dy)
        if not (result.ok and result.pvalue > 1e-6):
            failures.append((i, case.truth, result.redchi))
    assert not failures, failures
