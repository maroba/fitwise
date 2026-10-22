"""
RC discharge
============

A capacitor discharges through a known resistor. The voltage decays exponentially towards
a small offset of the measuring device. From the time constant τ = RC we obtain the
capacitance.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(3)
t = np.linspace(0, 50, 60)  # ms
voltage = 4.8 * np.exp(-t / 12.5) + 0.12 + rng.normal(0, 0.02, t.size)  # V

model = fw.models.exponential + fw.models.constant
result = fw.fit(model, t, voltage, yerr=0.02)
print(result)

# %%
# The ``Start:`` line of the report shows how the starting values were found: the offset
# ``c`` from the flat end of the curve (``edge baseline``), the decay from a log-linear fit
# of the rest.
fig, axes = result.plot()
axes[0].set_ylabel("U (V)")
axes[1].set_xlabel("t (ms)")

# %%
# Capacitance with R = 10.0 kΩ (assumed exact). τ is in ms, so C comes out in µF.
R = 10.0  # kΩ
tau = result.params["tau"]
print(f"C = {tau.value / R:.4f} ± {tau.stderr / R:.4f} µF")
