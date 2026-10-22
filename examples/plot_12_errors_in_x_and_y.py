"""
Errors in x and y: Ohm's law
============================

A resistor is characterized with a voltmeter and an ammeter. Both readings have
uncertainties, the current even more than the voltage. An ordinary fit treats x as exact and
underestimates the uncertainty of the resistance. With ``xerr``, fitwise uses orthogonal
distance regression, which accounts for the errors in both variables.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(12)
current_true = np.linspace(2, 20, 12)  # mA
R_true = 0.47  # kΩ, so U = R·I comes out in V
u_I, u_U = 0.4, 0.05  # instrument uncertainties: mA and V
current = current_true + rng.normal(0, u_I, current_true.size)
voltage = R_true * current_true + rng.normal(0, u_U, current_true.size)

ohm = "R*x"

# %%
# Ordinary fit, current treated as exact. The χ² warning says that the voltage errors
# alone cannot explain the scatter:
import warnings

with warnings.catch_warnings(record=True):
    ordinary = fw.fit(ohm, current, voltage, yerr=u_U)
print(ordinary)

# %%
# With ``xerr`` the fit is consistent, and the uncertainty of R is realistic:
odr = fw.fit(ohm, current, voltage, yerr=u_U, xerr=u_I)
print(odr)

# %%
# The plot shows error bars in both directions. The residuals are normalized by the effective
# uncertainty √(yerr² + (R·xerr)²).
fig, axes = odr.plot()
axes[0].set_ylabel("U (V)")
axes[1].set_xlabel("I (mA)")

# %%
for label, r in [("ordinary", ordinary), ("ODR", odr)]:
    R = r.params["R"]
    print(f"{label:8s} R = {1000 * R.value:.1f} ± {1000 * R.stderr:.1f} Ω   (true: 470 Ω)")
