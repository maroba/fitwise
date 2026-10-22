"""Starting values for user functions: multi-start with variable projection."""

import warnings

import numpy as np
import pytest
from free_models import CASES

import fitwise as fw
from fitwise._validate import validate_data
from fitwise.backends.base import FitProblem
from fitwise.backends.multistart import MultiStartBackend, conditionally_linear


def problem_for(func, x, y, names):
    model = fw.Model(func)
    data = validate_data(x, y, None, "raise")
    return FitProblem(
        model, data, dict.fromkeys(names, 1.0), {}, dict.fromkeys(names, (-np.inf, np.inf))
    )


class TestConditionalLinearity:
    x = np.linspace(400, 700, 100)

    def test_amplitude_and_offset_of_a_far_peak(self):
        def peak(x, a, mu, s, c):
            return a * np.exp(-((x - mu) ** 2) / (2 * s**2)) + c

        y = peak(self.x, 3, 550, 20, 1)
        problem = problem_for(peak, self.x, y, ["a", "mu", "s", "c"])
        # at the fallback start (mu = s = 1) the peak is exactly zero over the data; the check
        # must not mistake mu and s for linear parameters
        assert conditionally_linear(problem, np.random.default_rng(0)) == ["a", "c"]

    def test_products_are_not_jointly_linear(self):
        problem = problem_for(lambda x, a, b: a * b * x, self.x, self.x, ["a", "b"])
        assert conditionally_linear(problem, np.random.default_rng(0)) == ["a"]

    def test_nothing_linear(self):
        problem = problem_for(lambda x, k: np.exp(-k * x / 700), self.x, self.x, ["k"])
        assert conditionally_linear(problem, np.random.default_rng(0)) == []


class TestFit:
    def test_poorly_scaled_parameters_get_uncertainties(self, rng):
        """Regression: the covariance used to come from the first run's normalization."""
        temp = np.linspace(280, 380, 12)
        k = 2.6e11 * np.exp(-42_690 / (8.314 * temp))
        k = k * (1 + rng.normal(0, 0.01, temp.size))

        def arrhenius(t, A, Ea):  # noqa: N803
            return A * np.exp(-Ea / (8.314 * t))

        r = fw.fit(arrhenius, temp, k, yerr=0.01 * k)
        good = fw.fit(arrhenius, temp, k, yerr=0.01 * k, p0={"A": 2.6e11, "Ea": 42_690})
        assert r.ok
        np.testing.assert_allclose(r.params.as_array(), good.params.as_array(), rtol=1e-6)
        np.testing.assert_allclose(r.covariance, good.covariance, rtol=1e-3)

    def test_report_and_log(self, rng):
        x = np.linspace(400, 700, 150)
        y = 5 * np.exp(-((x - 613) ** 2) / (2 * 8.6**2)) + 1.9 + rng.normal(0, 0.05, x.size)

        def peak(x, a, mu, s, c):
            return a * np.exp(-((x - mu) ** 2) / (2 * s**2)) + c

        r = fw.fit(peak, x, y, yerr=0.05)
        assert r.ok
        assert r.params["mu"].value == pytest.approx(613, abs=0.5)
        assert r.method_reason == "auto: nonlinear, no starting values"
        assert r.method.endswith("with multi-start")
        assert set(r.guess_strategy.values()) == {"multi-start"}
        assert "linear parameters solved by projection: a, c" in r.solver_log[0]

    def test_given_starting_values_are_not_sampled(self):
        x = np.linspace(0, 10, 50)
        y = 3 * np.exp(-x / 2.5) + 0.5

        def decay(x, a, tau, c):
            return a * np.exp(-x / tau) + c

        r = fw.fit(decay, x, y, p0={"tau": 2.0})
        assert r.guess_strategy == {"a": "multi-start", "tau": "p0", "c": "multi-start"}
        assert r.initial_guess["tau"] == 2.0

    def test_signature_defaults_skip_the_multi_start(self):
        x = np.linspace(0, 10, 50)

        def decay(x, a=1.0, tau=1.0):
            return a * np.exp(-x / tau)

        r = fw.fit(decay, x, 3 * np.exp(-x / 2.5))
        assert r.method_reason == "auto: nonlinear"

    def test_explicit_multistart_and_reproducibility(self):
        x = np.linspace(0, 10, 200)
        y = 2 * np.sin(3.1 * x + 0.4)

        def sine(x, a, w, phi):
            return a * np.sin(w * x + phi)

        r1 = fw.fit(sine, x, y + 0.01 * np.cos(17 * x), method="multistart")
        r2 = fw.fit(sine, x, y + 0.01 * np.cos(17 * x), method="multistart")
        assert r1.method_reason == "requested"
        assert r1.params.values_dict() == r2.params.values_dict()
        assert abs(r1.params["w"].value) == pytest.approx(3.1, rel=1e-3)

    def test_no_candidate_refinable_falls_back(self, monkeypatch):
        backend = MultiStartBackend(["a"], n_candidates=3)
        x = np.linspace(1, 2, 10)
        problem = problem_for(lambda x, a: a * x, x, 2 * x, ["a"])
        monkeypatch.setattr("fitwise.backends.multistart._chi2", lambda *args: float("inf"))
        monkeypatch.setattr("fitwise.backends.multistart.conditionally_linear", lambda *args: [])
        solved = backend.solve(problem)
        assert any("fell back" in line for line in solved.log)
        assert solved.values["a"] == pytest.approx(2)


@pytest.mark.slow
def test_benchmark_catalog():
    """32 realistic models as plain functions, fitted without any starting values."""
    failures = []
    total = 0
    for case in CASES:
        rng = np.random.default_rng(2026)
        for i in range(8):
            x, y, dy, _ = case.sample(rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", fw.FitWarning)
                r = fw.fit(case.func, x, y, yerr=dy)
            total += 1
            if not (r.ok and r.pvalue > 1e-6):
                failures.append(f"{case.name}#{i}")
    assert len(failures) <= 0.03 * total, failures
