"""
Errors in x and y: Ohm's law with units
=======================================

A resistor is characterized with a voltmeter and an ammeter. Both readings have
uncertainties, the current even more than the voltage. An ordinary fit treats x as exact and
underestimates the uncertainty of the resistance. With ``xerr``, fitwise uses orthogonal
distance regression, which accounts for the errors in both variables.

The data carry units (`pint <https://pint.readthedocs.io>`_), so the resistance comes out
with its unit and can be converted directly.
"""

# %%
import warnings

import numpy as np
import pint

import fitwise as fw

u = pint.UnitRegistry()
rng = np.random.default_rng(12)
current_true = np.linspace(2, 20, 12)  # mA
current = (current_true + rng.normal(0, 0.4, current_true.size)) * u.mA
voltage = (0.47 * current_true + rng.normal(0, 0.05, current_true.size)) * u.V
u_I, u_U = 0.4 * u.mA, 50 * u.mV  # instrument uncertainties

# %%
# Ordinary fit, current treated as exact. The χ² warning says that the voltage errors
# alone cannot explain the scatter. The slope of the built-in ``linear`` model gets the unit
# V/mA automatically:
with warnings.catch_warnings(record=True):
    ordinary = fw.fit(fw.models.linear, current, voltage, yerr=u_U, fixed={"intercept": 0})
print(ordinary)

# %%
# With ``xerr`` the fit is consistent, and the uncertainty of R is realistic:
odr = fw.fit(fw.models.linear, current, voltage, yerr=u_U, xerr=u_I, fixed={"intercept": 0})
print(odr)

# %%
# The plot shows error bars in both directions and labels the axes with the units. The
# residuals are normalized by the effective uncertainty √(yerr² + (R·xerr)²).
fig, axes = odr.plot()

# %%
# ``measurement`` gives value ± error with unit, ready for conversion:
for label, r in [("ordinary", ordinary), ("ODR", odr)]:
    print(f"{label:8s} R = {r.params['slope'].measurement.to('ohm'):~P}   (true: 470 Ω)")
