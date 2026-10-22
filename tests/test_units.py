"""Fits with pint quantities."""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pint
import pytest

import fitwise as fw
from fitwise.models import constant, exponential, gaussian, linear, polynomial, powerlaw

u = pint.UnitRegistry()


@pytest.fixture
def ohm(rng):
    current = np.linspace(2, 20, 12) * u.mA
    voltage = (0.47 * current.magnitude + 0.3 + rng.normal(0, 0.05, 12)) * u.V
    return current, voltage


class TestParameterUnits:
    def test_built_in_models(self, ohm):
        current, voltage = ohm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)  # only the units matter here
            r = fw.fit(polynomial(2) + gaussian, current, voltage)
        units = {n: f"{p.unit:~P}" for n, p in r.params.items()}
        assert units == {
            "c0": "V",
            "c1": "V/mA",
            "c2": "V/mA²",
            "amp": "V",
            "cen": "mA",
            "sigma": "mA",
        }

    def test_prefixes_and_dimensionless(self, ohm):
        current, voltage = ohm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(gaussian + gaussian, current, voltage)
            assert r.params["g2_cen"].unit == u.mA
            p = fw.fit(powerlaw, current, voltage)
        assert p.params["amp"].unit is None  # unit depends on the exponent
        assert p.params["exponent"].unit == u.dimensionless

    def test_products_declare_no_units(self, ohm):
        current, voltage = ohm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)
            r = fw.fit(linear * constant, current, voltage, fixed={"c": 1.0})
        assert all(p.unit is None for p in r.params.values())

    def test_given_units_for_expressions(self, rng):
        t = np.linspace(0, 50, 40) * u.ms
        v = (4.8 * np.exp(-t.magnitude / 12.5) + rng.normal(0, 0.02, 40)) * u.V
        r = fw.fit("a*exp(-x/tau)", t, v, yerr=0.02 * u.V, units={"tau": "ms", "a": u.V})
        assert r.params["tau"].unit == u.ms
        assert r.params["tau"].measurement.to("s").value.magnitude == pytest.approx(
            0.0125, rel=0.01
        )


class TestConversion:
    def test_errors_are_converted(self, ohm):
        current, voltage = ohm
        a = fw.fit(linear, current, voltage, yerr=50 * u.mV)
        b = fw.fit(linear, current.magnitude, voltage.magnitude, yerr=0.05)
        c = fw.fit(linear, current, voltage, yerr=0.05)  # plain numbers: unit of y
        np.testing.assert_allclose(a.params.as_array(), b.params.as_array())
        np.testing.assert_allclose(a.covariance, b.covariance)
        np.testing.assert_allclose(c.covariance, b.covariance)

    def test_xerr_is_converted(self, ohm):
        current, voltage = ohm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fw.FitWarning)  # only the conversion matters
            a = fw.fit(linear, current, voltage, yerr=0.05 * u.V, xerr=0.0004 * u.A)
            b = fw.fit(linear, current.magnitude, voltage.magnitude, yerr=0.05, xerr=0.4)
        np.testing.assert_allclose(a.params.as_array(), b.params.as_array(), rtol=1e-9)

    def test_quantities_in_p0_bounds_fixed(self, rng):
        t = np.linspace(0, 50, 60) * u.ms
        v = (4.8 * np.exp(-t.magnitude / 12.5) + 0.12 + rng.normal(0, 0.02, 60)) * u.V
        model = exponential + constant
        r = fw.fit(model, t, v, yerr=0.02, p0={"tau": 0.01 * u.s}, bounds={"tau": (1 * u.ms, None)})
        assert r.initial_guess["tau"] == pytest.approx(10.0)
        assert r.params["tau"].bounds == (1.0, np.inf)
        held = fw.fit(model, t, v, yerr=0.02, fixed={"c": 120 * u.mV})
        assert held.params["c"].value == pytest.approx(0.12)

    def test_eval_and_bands_with_quantities(self, ohm):
        current, voltage = ohm
        r = fw.fit(linear, current, voltage, yerr=0.05)
        out = r.eval([0.01] * u.A)
        assert out.units == u.V
        assert out.to("mV").magnitude[0] == pytest.approx(r.eval(np.array([10.0]))[0] * 1000)
        band = r.confidence_band([10] * u.mA)
        np.testing.assert_allclose(band.center, r.eval(np.array([10.0])))


class TestErrors:
    def test_yerr_unit_without_y_unit(self, ohm):
        current, voltage = ohm
        with pytest.raises(ValueError, match="yerr has a unit, but y has none"):
            fw.fit(linear, current, voltage.magnitude, yerr=0.05 * u.V)

    def test_incompatible_units(self, ohm):
        current, voltage = ohm
        with pytest.raises(ValueError, match="incompatible"):
            fw.fit(linear, current, voltage, yerr=0.05 * u.s)

    def test_unknown_parameter_unit(self, rng):
        t = np.linspace(0, 50, 40) * u.ms
        v = 4.8 * np.exp(-t.magnitude / 12.5) * u.V
        with pytest.raises(ValueError, match="declare it with fit"):
            fw.fit("a*exp(-x/tau)", t, v, p0={"tau": 10 * u.ms})
        with pytest.raises(ValueError, match="Unknown parameter 'tua'"):
            fw.fit("a*exp(-x/tau)", t, v, units={"tua": "ms"})

    def test_quantity_and_measurement_need_unit_and_error(self, rng):
        x = np.linspace(0, 1, 10)
        r = fw.fit(linear, x, 2 * x + 1 + rng.normal(0, 0.01, 10))
        with pytest.raises(ValueError, match="unit"):
            r.params["slope"].quantity  # noqa: B018
        held = fw.fit(linear, x * u.s, (2 * x + 1) * u.m, fixed={"intercept": 1.0})
        with pytest.raises(ValueError, match="no standard error"):
            held.params["intercept"].measurement  # noqa: B018


class TestOutput:
    def test_report_has_unit_column(self, ohm):
        current, voltage = ohm
        text = str(fw.fit(linear, current, voltage, yerr=0.05))
        header = next(line for line in text.splitlines() if line.startswith("Parameter"))
        assert "Unit" in header
        assert "V/mA" in text
        x = np.linspace(0, 1, 10)
        plain = str(fw.fit(linear, x, 2 * x + 1 + 0.01 * np.sin(9 * x)))
        assert "Unit" not in plain

    def test_plot_labels_and_export(self, ohm):
        current, voltage = ohm
        r = fw.fit(linear, current, voltage, yerr=0.05)
        fig, axes = r.plot()
        assert axes[0].get_ylabel() == "y (V)"
        assert axes[1].get_xlabel() == "x (mA)"
        plt.close(fig)
        d = r.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert d["data"]["x_unit"] == "mA"
        assert d["data"]["y_unit"] == "V"
        assert d["parameters"]["slope"]["unit"] == "V/mA"
