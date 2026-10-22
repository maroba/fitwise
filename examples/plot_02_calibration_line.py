"""
Calibration line
================

A thermocouple voltage is calibrated against reference temperatures. The model is a
straight line; each voltage reading has its own uncertainty.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(2)
temperature = np.array([0.0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])  # °C
u_volt = 0.010 + 0.0002 * temperature  # mV, larger at high temperatures
voltage = 0.0410 * temperature - 0.02 + rng.normal(0, u_volt)

result = fw.fit(fw.models.linear, temperature, voltage, yerr=u_volt)
print(result)

# %%
# The slope is the sensitivity of the thermocouple in mV/°C. Note the strong negative
# correlation between slope and intercept, listed in the report: both are estimated from the
# same points, so their errors are not independent.
fig, axes = result.plot(show_params=True)
axes[0].set_ylabel("voltage (mV)")
axes[1].set_xlabel("temperature (°C)")

# %%
# Because the uncertainties were given, χ²/ndf tests whether they are realistic. A value near
# 1 means the line describes the data within their errors.
print(f"χ²/ndf = {result.redchi:.2f}, p = {result.pvalue:.2f}")
