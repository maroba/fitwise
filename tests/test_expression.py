"""Models written as text."""

import numpy as np
import pytest

import fitwise as fw
from fitwise.expression import ExpressionModel, parse_expression


class TestParsing:
    @pytest.mark.parametrize(
        ("expr", "params"),
        [
            ("a*exp(-x/tau) + c", ["a", "tau", "c"]),
            ("c + a*exp(-x/tau)", ["c", "a", "tau"]),
            ("A*sin(2*pi*f*x + phi)", ["A", "f", "phi"]),
            ("a*x + a*x**2 + b", ["a", "b"]),
            ("amp / (1 + ((x - x0)/w)^2)", ["amp", "x0", "w"]),
        ],
    )
    def test_parameters_in_order_of_appearance(self, expr, params):
        _, found = parse_expression(expr)
        assert found == params

    @pytest.mark.parametrize(
        ("expr", "message"),
        [
            ("a.__class__", "Attribute is not allowed"),
            ("__import__('os').system('true')", "only plain function calls"),
            ("__import__('os')*x", "unknown function '__import__'"),
            ("x[0]*a", "Subscript is not allowed"),
            ("(lambda: 1)()*a*x", "only plain function calls"),
            ("[a for a in x]", "ListComp is not allowed"),
            ("exp(x, out=a)", "only plain function calls"),
            ("a*foo(x)", "unknown function 'foo'"),
            ("a*exp", "is a function and must be called"),
            ("'s'*a*x", "is not a number"),
            ("True*a*x", "is not a number"),
            ("a*x if a else x", "IfExp is not allowed"),
            ("a*x % 2", "Mod is not allowed"),
            ("not a*x", "Not is not allowed"),
            ("a < x", "Compare is not allowed"),
            ("a*b", "does not contain the independent variable 'x'"),
            ("2*x", "has no parameters"),
            ("a*x +", "invalid expression"),
            ("", "non-empty string"),
        ],
    )
    def test_rejected(self, expr, message):
        with pytest.raises(ValueError, match=message):
            parse_expression(expr)

    def test_keyword_parameter(self):
        with pytest.raises(ValueError, match="invalid expression"):
            parse_expression("lambda*x")

    def test_caret_is_power_with_power_precedence(self):
        """Regression: as an AST operator, ^ binds weaker than + and *."""
        m = ExpressionModel("a + b*x^2")
        assert m(np.array([3.0]), a=1.0, b=2.0)[0] == 19.0
        assert ExpressionModel("2^x*a")(np.array([3.0]), a=1.0)[0] == 8.0


class TestModel:
    def test_signature_and_names(self):
        m = ExpressionModel("a*exp(-t/tau)", independent="t")
        assert m.signature() == "y = a*exp(-t/tau)"
        assert m.param_names == ("a", "tau")
        assert m.formula == "a*exp(-t/tau)"
        assert m.independent == "t"

    def test_functions_and_constants(self):
        m = ExpressionModel("a*erf(x) + b*ln(x) + c*sqrt(x) + d*cos(pi*x)")
        x = np.array([1.0])
        expected = 1 * 0.8427007929497149 + 0 + 3 * 1 + 4 * np.cos(np.pi)
        assert m(x, 1, 2, 3, 4)[0] == pytest.approx(expected)

    def test_no_builtins_available(self):
        with pytest.raises(ValueError, match="unknown function 'open'"):
            ExpressionModel("open(x)*a")

    def test_composes_with_built_in_models(self):
        m = ExpressionModel("k*x") + fw.models.constant
        assert m.param_names == ("k", "c")


class TestFit:
    def test_string_in_fit(self, rng):
        x = np.linspace(0, 10, 60)
        y = 3 * np.exp(-x / 2.5) + 0.4 + rng.normal(0, 0.02, x.size)
        r = fw.fit("a*exp(-x/tau) + c", x, y, yerr=0.02)
        assert r.ok
        assert r.model.signature() == "y = a*exp(-x/tau) + c"
        assert str(r).splitlines()[0] == "Model:      y = a*exp(-x/tau) + c"
        np.testing.assert_allclose(r.params.as_array(), [3, 2.5, 0.4], rtol=0.05)

    def test_linear_expression_is_solved_directly(self, rng):
        x = np.linspace(0, 10, 60)
        r = fw.fit("a + b*x^2", x, 1 + 0.5 * x**2 + rng.normal(0, 0.1, x.size), yerr=0.1)
        assert r.method == "Linear least squares (direct)"
        np.testing.assert_allclose(r.params.as_array(), [1, 0.5], atol=0.05)
