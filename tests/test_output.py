"""Report, plot and serialization."""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fitwise as fw
from fitwise.models import gaussian, linear
from fitwise.parameters import Parameter, Parameters
from fitwise.report import format_measurement
from fitwise.serialize import data_hash


@pytest.fixture
def result(peak_data):
    x, y, dy = peak_data
    return fw.fit(gaussian + linear, x, y, yerr=dy, bounds={"sigma": (0, None)})


class TestFormatMeasurement:
    @pytest.mark.parametrize(
        ("value", "error", "expected"),
        [
            (1.51234, 0.01937, ("1.512", "0.019")),
            (2.331, 0.021, ("2.331", "0.021")),
            (-0.0031, 0.00094, ("-0.00310", "0.00094")),
            (56789.0, 1234.0, ("56800", "1200")),
            (0.0, 0.5, ("0.00", "0.50")),
            (1.0e7, 2.5e5, ("1.000e+07", "0.025e+07")),
            (-1.42626e-6, 5.567e-8, ("-1.426e-06", "0.056e-06")),
            (3e-9, 5e-6, ("0.0e-06", "5.0e-06")),
            (2.5, None, ("2.5", "–")),
            (2.5, 0.0, ("2.5", "–")),
            (2.5, math.inf, ("2.5", "–")),
            (math.nan, 1.0, ("nan", "–")),
        ],
    )
    def test_cases(self, value, error, expected):
        assert format_measurement(value, error) == expected


class TestReport:
    def test_layout(self, result):
        text = str(result)
        lines = text.splitlines()
        assert lines[0] == "Model:      gaussian(x; amp, cen, sigma) + linear(x; slope, intercept)"
        assert lines[1].startswith("Method:     Trust Region Reflective")
        assert lines[2].startswith("Converged:  yes (")
        assert lines[3] == "Data:       128 points, yerr given (absolute)"
        assert (
            lines[4]
            == "Start:      amp, cen, sigma: peak heuristic · slope, intercept: edge baseline"
        )
        assert lines[6].split() == ["Parameter", "Value", "±", "Std.Err", "Rel.", "Bounds"]
        assert lines[9].endswith("(0, ∞)")
        assert any(line.startswith("χ²/ndf = ") for line in lines)
        assert lines[-1] == "Warnings:   none"

    def test_repr(self, result):
        assert repr(result).startswith("<FitResult gaussian+linear: ok, χ²/ndf=")

    def test_without_yerr(self, peak_data):
        x, y, _ = peak_data
        text = str(fw.fit(gaussian, x, y - 0.1 + 0.003 * x))
        assert "no yerr (uncertainties from residual scatter)" in text
        assert "σ_res = " in text

    def test_r2_digits(self):
        from fitwise.report import _r2_text

        assert _r2_text(0.5) == "0.5000"
        assert _r2_text(0.9999991) == "0.99999910"
        assert _r2_text(1.0) == "1.0000"
        assert _r2_text(math.nan) == "–"

    def test_bound_formatting(self):
        from fitwise.report import _format_bound

        assert [_format_bound(b) for b in (-math.inf, math.inf, 0.5)] == ["-∞", "∞", "0.5"]


class TestParameters:
    def test_access(self, result):
        p = result.params
        assert p["sigma"] is p.sigma
        assert list(p) == ["amp", "cen", "sigma", "slope", "intercept"]
        assert len(p) == 5
        assert "sigma" in dir(p)
        assert float(p.amp) == p.amp.value
        assert p.amp.relative_error == pytest.approx(p.amp.stderr / abs(p.amp.value))
        assert set(p.errors_dict()) == set(p.values_dict())
        assert repr(p).startswith("Parameters([Parameter('amp', value=")
        with pytest.raises(KeyError, match="available"):
            p["nope"]
        with pytest.raises(AttributeError):
            p.nope  # noqa: B018
        with pytest.raises(AttributeError):
            p._private  # noqa: B018

    def test_repr_variants(self):
        assert (
            repr(Parameter("a", 1.0, None, 1.0, "p0", fixed=True))
            == "Parameter('a', value=1, fixed)"
        )
        assert repr(Parameter("a", 1.0, None, 1.0, "p0")) == "Parameter('a', value=1, stderr=None)"
        assert Parameter("a", 0.0, 1.0, 1.0, "p0").relative_error is None
        assert Parameters([]).as_array().shape == (0,)

    def test_result_accessors(self, result):
        assert result.x is result.data.x
        assert result.yerr is result.data.yerr
        assert result.initial_guess.keys() == result.guess_strategy.keys()
        np.testing.assert_allclose(result.residuals, result.y - result.eval())
        assert result.eval(np.array([0.21])).shape == (1,)


class TestPlot:
    def test_with_residuals(self, result):
        fig, axes = result.plot(show_params=True)
        assert len(axes) == 2
        assert axes[1].get_ylabel() == "(y − f)/σ"
        assert any("amp = " in t.get_text() for t in axes[0].get_legend().get_texts())
        plt.close(fig)

    def test_without_residuals_and_existing_axes(self, result, peak_data):
        fig, axes = result.plot(residuals=False)
        assert len(axes) == 1
        plt.close(fig)
        fig, ax = plt.subplots()
        fig2, axes = result.plot(ax=ax)
        assert fig2 is fig
        assert axes == [ax]
        plt.close(fig)
        x, y, _ = peak_data
        fig, axes = fw.fit(gaussian + linear, x, y).plot()
        assert axes[1].get_ylabel() == "y − f"
        plt.close(fig)

    def test_multidimensional_x_rejected(self):
        xy = np.random.default_rng(0).uniform(size=(20, 2))
        r = fw.fit(lambda p, a: a * p[:, 0], xy, 2 * xy[:, 0] + 0.01 * xy[:, 1])
        with pytest.raises(ValueError, match="one-dimensional"):
            r.plot()

    def test_missing_matplotlib(self, result, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match=r"fitwise\[plot\]"):
            result.plot()


class TestSerialize:
    def test_json_roundtrip(self, result):
        d = result.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert d["schema_version"] == 1
        assert d["parameters"]["sigma"]["bounds"] == [0.0, "inf"]
        assert d["options"]["bounds"] == {"sigma": [0.0, "inf"]}
        assert d["covariance"]["names"] == list(result.free_names)
        assert d["statistics"]["ndf"] == 123
        assert d["warnings"] == []
        assert "x" not in d["data"]
        assert d["versions"]["fitwise"] == fw.__version__

    def test_include_data_and_hash(self, result):
        d = result.to_dict(include_data=True)
        assert len(d["data"]["x"]) == 128
        assert d["data"]["sha256"] == data_hash(result.x, result.y, result.yerr)
        assert data_hash(result.x, result.y, None) != data_hash(result.x, result.y, result.yerr)

    def test_special_values(self):
        x = np.array([0.0, 1.0])
        with pytest.warns(fw.FitWarning):
            d = fw.fit(linear, x, np.array([1.0, 3.0])).to_dict()
        assert d["covariance"]["matrix"] is None
        assert d["statistics"]["redchi"] is None
        assert d["warnings"][0]["type"] == "SingularCovarianceWarning"
        from fitwise.serialize import _float

        assert [_float(v) for v in (math.nan, -math.inf)] == ["nan", "-inf"]
