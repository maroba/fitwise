"""Bootstrap and profile-likelihood intervals."""

import warnings

import numpy as np
import pytest

import fitwise as fw
from fitwise.intervals import ONE_SIGMA, Interval
from fitwise.models import constant, exponential, linear


@pytest.fixture
def line(rng):
    x = np.linspace(0, 5, 30)
    return fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size), yerr=0.2)


@pytest.fixture
def short_decay(rng):
    x = np.linspace(0, 3, 8)
    return fw.fit(exponential, x, 5 * np.exp(-x / 0.8) + rng.normal(0, 0.15, x.size), yerr=0.15)


class TestProfile:
    @pytest.mark.parametrize("with_yerr", [True, False])
    def test_linear_model_gives_plus_minus_sigma(self, rng, with_yerr):
        x = np.linspace(0, 5, 30)
        y = 2 * x + 1 + rng.normal(0, 0.2, x.size)
        r = fw.fit(linear, x, y, yerr=0.2 if with_yerr else None)
        pr = r.profile()
        for name, iv in pr.intervals.items():
            assert iv.minus == pytest.approx(r.params[name].stderr, rel=1e-6)
            assert iv.plus == pytest.approx(r.params[name].stderr, rel=1e-6)
        assert pr.delta_chi2 == pytest.approx(1.0 if with_yerr else r.redchi)

    def test_nonlinear_model_is_asymmetric(self, short_decay):
        iv = short_decay.profile(["tau"])["tau"]
        assert iv.plus > iv.minus
        assert iv.minus == pytest.approx(short_decay.params["tau"].stderr, rel=0.1)

    def test_level(self, line):
        iv95 = line.profile(["slope"], level=0.95)["slope"]
        assert iv95.minus == pytest.approx(1.959964 * line.params["slope"].stderr, rel=1e-5)
        assert iv95.level == 0.95

    def test_interval_stops_at_bound(self, line):
        p = line.params["intercept"]
        bound = p.value - 0.5 * p.stderr  # inside the 1σ range, but below the value
        r = fw.fit(linear, line.x, line.y, yerr=0.2, bounds={"intercept": (bound, None)})
        iv = r.profile(["intercept"])["intercept"]
        assert iv.lower == bound
        assert iv.plus == pytest.approx(p.stderr, rel=1e-6)

    def test_open_interval(self):
        x = np.linspace(0, 5, 20)
        y = 1.0 + 0.001 * np.sin(7 * x)  # flat: a fast decay cannot be excluded

        def decay(x, c, a, k):
            return c + a * np.exp(-k * x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(
                decay, x, y, yerr=0.01, p0={"c": 1.0, "a": 0.5, "k": 1.0}, bounds={"k": (0, None)}
            )
            iv = r.profile(["k"])["k"]
        assert iv.upper == np.inf

    def test_values_where_the_model_fails_are_excluded(self):
        x = np.linspace(1, 5, 20)

        def logmodel(x, a, c):
            return a * np.log(x - c)

        y = 2 * np.log(x - 0.5) + 0.01 * np.sin(9 * x)
        r = fw.fit(logmodel, x, y, yerr=0.01, p0={"a": 2, "c": 0.5})
        iv = r.profile(["c"])["c"]  # c >= 1 makes log(x - c) undefined at x = 1
        assert iv.lower < r.params["c"].value < iv.upper < 1

    def test_errors_and_text(self, line):
        with pytest.raises(ValueError, match="not a free parameter"):
            line.profile(["nope"])
        with pytest.raises(ValueError, match="level"):
            line.profile(level=1.0)
        text = str(line.profile())
        assert text.startswith("Profile likelihood: 68.3 % intervals (Δχ² = 1)")
        assert "slope" in text
        x = np.linspace(0, 5, 30)
        robust = fw.fit(linear, x, 2 * x + 1, yerr=0.2, loss="huber")
        with pytest.raises(NotImplementedError, match="robust"):
            robust.profile()

    def test_with_errors_in_x(self):
        from test_odr import XERR, YERR, X, Y

        r = fw.fit(linear, X, Y, yerr=YERR, xerr=XERR)
        iv = r.profile(["slope"])["slope"]
        assert iv.minus == pytest.approx(r.params["slope"].stderr, rel=0.1)
        assert iv.plus == pytest.approx(r.params["slope"].stderr, rel=0.1)

    def test_interval_tuple(self):
        iv = Interval(1.0, 0.8, 1.5, ONE_SIGMA)
        assert iv.minus == pytest.approx(0.2)
        assert iv.plus == pytest.approx(0.5)


class TestBootstrap:
    def test_parametric_matches_covariance(self, line):
        b = line.bootstrap(1500, seed=3)
        assert b.method == "parametric"
        for name in line.free_names:
            assert b.std[name] == pytest.approx(line.params[name].stderr, rel=0.08)
        np.testing.assert_allclose(b.covariance, line.covariance, rtol=0.2, atol=1e-6)

    def test_residuals_reflect_the_observed_scatter(self, line):
        b = line.bootstrap(1500, method="residuals", seed=3)
        expected = np.sqrt(line.redchi)
        for name in line.free_names:
            assert b.std[name] / line.params[name].stderr == pytest.approx(expected, rel=0.1)

    def test_auto_without_yerr_uses_residuals(self, rng):
        x = np.linspace(0, 5, 30)
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size))
        b = r.bootstrap(200)
        assert b.method == "residuals"
        assert b.std["slope"] == pytest.approx(r.params["slope"].stderr, rel=0.2)

    def test_cases_and_intervals(self, line):
        b = line.bootstrap(400, method="cases")
        iv = b.interval("slope")
        assert iv.lower < line.params["slope"].value < iv.upper
        assert b.samples.shape == (400 - b.n_failed, 2)
        text = str(b)
        assert text.startswith("Bootstrap (cases): ")
        assert "σ (boot)" in text

    def test_reproducible(self, short_decay):
        a = short_decay.bootstrap(50, seed=7).samples
        b = short_decay.bootstrap(50, seed=7).samples
        np.testing.assert_array_equal(a, b)

    def test_failed_refits_are_counted(self, line, monkeypatch):
        import fitwise.intervals as iv

        real = iv._refit
        calls = {"n": 0}

        def flaky(result, **kwargs):
            calls["n"] += 1
            out = real(result, **kwargs)
            if calls["n"] % 5 == 0:
                from dataclasses import replace

                return replace(out, converged=False)
            return out

        monkeypatch.setattr(iv, "_refit", flaky)
        b = line.bootstrap(50)
        assert b.n_failed == 10
        assert "10 failed and excluded" in str(b)

    def test_errors(self, line, rng):
        with pytest.raises(ValueError, match="at least 2"):
            line.bootstrap(1)
        with pytest.raises(ValueError, match="unknown method"):
            line.bootstrap(10, method="jackknife")
        x = np.linspace(0, 5, 30)
        no_yerr = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.2, x.size))
        with pytest.raises(ValueError, match="needs yerr"):
            no_yerr.bootstrap(10, method="parametric")
        robust = fw.fit(linear, x, 2 * x + 1, yerr=0.2, loss="huber")
        with pytest.raises(NotImplementedError, match="ordinary least-squares"):
            robust.bootstrap(10)


@pytest.mark.slow
def test_profile_coverage():
    """Monte Carlo: 68.3 % profile intervals of a short decay cover the truth about 68 %."""
    x = np.linspace(0, 3, 8)
    hits, n = 0, 200
    for seed in range(n):
        rng = np.random.default_rng(seed)
        y = 5 * np.exp(-x / 0.8) + rng.normal(0, 0.15, x.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(exponential + constant, x, y, yerr=0.15, fixed={"c": 0.0})
        iv = r.profile(["tau"])["tau"]
        hits += iv.lower <= 0.8 <= iv.upper
    assert abs(hits / n - ONE_SIGMA) < 0.07
