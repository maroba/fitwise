"""
Power law: vacuum diode
=======================

In a vacuum diode, the space-charge limited anode current follows the Child–Langmuir law
I = k · U^(3/2). Fitting a general power law tests whether the exponent really is 3/2.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(10)
voltage = np.linspace(10, 200, 25)  # V
current = 0.018 * voltage**1.5 * (1 + rng.normal(0, 0.02, voltage.size))  # mA
dI = 0.02 * current

result = fw.fit(fw.models.powerlaw, voltage, current, yerr=dI)
print(result)

# %%
# Is the exponent compatible with 3/2?
n = result.params["exponent"]
deviation = (n.value - 1.5) / n.stderr
print(f"exponent = {n.value:.4f} ± {n.stderr:.4f}, deviation from 1.5: {deviation:+.1f} σ")

# %%
# If theory fixes the exponent, fix it and fit only the prefactor:
fixed = fw.fit(fw.models.powerlaw, voltage, current, yerr=dI, fixed={"exponent": 1.5})
print(fixed)

# %%
fig, axes = result.plot()
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_ylabel("I (mA)")
axes[1].set_xlabel("U (V)")
