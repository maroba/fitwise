import numpy as np
import pytest

import fitwise as fw
from fitwise.model import Model, as_model
from fitwise.models import constant, gaussian, linear, lorentzian


def decay(t, amp, tau=2.0):
    return amp * np.exp(-t / tau)


class TestIntrospection:
    def test_names_and_defaults_from_signature(self):
        m = Model(decay)
        assert m.name == "decay"
        assert m.param_names == ("amp", "tau")
        assert m.defaults == {"tau": 2.0}

    def test_lambda_gets_generic_name(self):
        assert Model(lambda x, a: a * x).name == "model"

    def test_explicit_param_names_for_varargs(self):
        m = Model(lambda x, *p: p[0] + p[1] * x, param_names=["a", "b"])
        assert m(np.array([0.0, 1.0]), 1, 2).tolist() == [1.0, 3.0]

    def test_varargs_without_names_rejected(self):
        with pytest.raises(TypeError, match="param_names"):
            Model(lambda x, *p: x)

    def test_keyword_only_rejected(self):
        def f(x, *, a):
            return a * x

        with pytest.raises(TypeError, match="keyword-only"):
            Model(f)

    @pytest.mark.parametrize("func", [lambda: 1, lambda x: x])
    def test_needs_parameters(self, func):
        with pytest.raises(TypeError):
            Model(func)

    def test_non_numeric_default_rejected(self):
        with pytest.raises(TypeError, match="must be a number"):
            Model(lambda x, a="big": x)

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Model(lambda x, *p: x, param_names=["a", "a"])

    def test_invalid_names_rejected(self):
        with pytest.raises(ValueError, match="identifier"):
            Model(lambda x, *p: x, param_names=["lambda"])

    def test_not_callable(self):
        with pytest.raises(TypeError, match="callable"):
            Model(42)  # type: ignore[arg-type]

    def test_unknown_symmetric(self):
        with pytest.raises(ValueError, match="symmetric"):
            Model(decay, symmetric=["sigma"])


class TestEvaluation:
    def test_call_positional_keyword_and_default(self):
        m = Model(decay)
        t = np.array([0.0, 2.0])
        np.testing.assert_allclose(m(t, 3.0), 3 * np.exp(-t / 2))
        np.testing.assert_allclose(m(t, amp=3.0, tau=1.0), 3 * np.exp(-t))

    @pytest.mark.parametrize(
        ("args", "kwargs", "match"),
        [
            ((1, 2, 3), {}, "takes 2"),
            ((), {"nope": 1}, "no parameter"),
            ((1,), {"amp": 1}, "given twice"),
            ((), {}, "missing"),
        ],
    )
    def test_call_errors(self, args, kwargs, match):
        with pytest.raises(TypeError, match=match):
            Model(decay)(np.zeros(2), *args, **kwargs)


class TestComposition:
    def test_sum_keeps_unique_names(self):
        m = gaussian + linear
        assert m.param_names == ("amp", "cen", "sigma", "slope", "intercept")
        assert m.linear is False
        assert m.signature() == "gaussian(x; amp, cen, sigma) + linear(x; slope, intercept)"

    def test_clashing_names_get_numbered_prefixes(self):
        m = gaussian + gaussian + gaussian
        assert m.param_names == tuple(
            f"g{i}_{n}" for i in (1, 2, 3) for n in ("amp", "cen", "sigma")
        )

    def test_clash_between_different_models_uses_abbrev(self):
        m = gaussian + lorentzian
        assert m.param_names == ("g_amp", "g_cen", "g_sigma", "l_amp", "l_cen", "l_gamma")

    def test_builtin_models_are_not_mutated(self):
        _ = gaussian + gaussian
        assert gaussian.param_names == ("amp", "cen", "sigma")

    def test_explicit_prefix_is_kept(self):
        m = gaussian.with_prefix("peak_") + gaussian
        assert m.param_names == ("peak_amp", "peak_cen", "peak_sigma", "amp", "cen", "sigma")

    def test_explicit_prefix_clash_is_an_error(self):
        with pytest.raises(ValueError, match="distinct prefixes"):
            gaussian.with_prefix("p_") + gaussian.with_prefix("p_")

    def test_invalid_prefix(self):
        with pytest.raises(ValueError, match="identifier"):
            gaussian.with_prefix("1x")

    def test_composite_cannot_be_prefixed(self):
        with pytest.raises(TypeError):
            (gaussian + linear).with_prefix("a_")

    def test_eval_sum_and_product(self):
        x = np.linspace(-1, 1, 5)
        p = {"amp": 2.0, "cen": 0.0, "sigma": 1.0, "slope": 1.0, "intercept": 0.5}
        g = gaussian.eval(x, p)
        np.testing.assert_allclose((gaussian + linear).eval(x, p), g + x + 0.5)
        np.testing.assert_allclose((gaussian * linear).eval(x, p), g * (x + 0.5))

    def test_function_on_either_side(self):
        m = constant + (lambda x, a: a * x)
        assert m.param_names == ("c", "a")
        m2 = (lambda x, a: a * x) * constant
        assert m2.op == "*"
        assert m2.param_names == ("a", "c")

    def test_mixed_signature_has_parentheses(self):
        m = (gaussian + constant) * linear
        assert m.signature().startswith("(gaussian")
        assert m.linear is False

    def test_linearity(self):
        assert (constant + linear).linear is True
        assert (constant + Model(lambda x, a: a * x)).linear is None

    def test_repr(self):
        assert repr(gaussian + linear).startswith("CompositeModel(gaussian(x; ")
        assert repr(gaussian) == "Model(gaussian(x; amp, cen, sigma))"


class TestGuessAndCanonicalize:
    def test_guess_public(self):
        x = np.linspace(-3, 3, 61)
        guess = gaussian.guess(x, gaussian(x, 2.0, 0.5, 0.7))
        assert guess["amp"] == pytest.approx(2.0, rel=0.01)
        assert guess["cen"] == pytest.approx(0.5, abs=0.05)
        assert guess["sigma"] == pytest.approx(0.7, rel=0.1)

    def test_model_without_guess(self):
        assert Model(decay).guess(np.zeros(3), np.zeros(3)) == {}

    def test_failing_guess_returns_empty(self):
        def bad(x, y):
            raise ValueError("no")

        assert Model(decay, guess=bad).guess(np.zeros(3), np.zeros(3)) == {}

    def test_non_finite_guesses_dropped(self):
        m = Model(decay, guess=lambda x, y: {"amp": np.nan, "tau": 1.0})
        assert m.guess(np.zeros(3), np.zeros(3)) == {"tau": 1.0}

    def test_canonicalize_flips_symmetric(self):
        m = gaussian + gaussian
        values, signs = m.canonicalize(
            {"g1_amp": 1, "g1_cen": 0, "g1_sigma": -2, "g2_amp": 1, "g2_cen": 0, "g2_sigma": 3}
        )
        assert values["g1_sigma"] == 2
        assert signs == {
            "g1_amp": 1,
            "g1_cen": 1,
            "g1_sigma": -1,
            "g2_amp": 1,
            "g2_cen": 1,
            "g2_sigma": 1,
        }

    def test_product_guess_uses_leaves(self):
        x = np.linspace(0, 5, 50)
        guess = (fw.models.exponential * constant).guess(x, np.exp(-x))
        assert set(guess) == {"amp", "tau", "c"}


def test_as_model():
    assert as_model(gaussian) is gaussian
    assert as_model(decay).param_names == ("amp", "tau")
    assert as_model("a*x").param_names == ("a",)
    with pytest.raises(TypeError):
        as_model(3)  # type: ignore[arg-type]
