"""
Outliers: a robust fit
======================

A temperature sensor cools down in air, and the logger records a few corrupted readings. An
ordinary least-squares fit is pulled towards them; ``loss="huber"`` downweights them. The
model is written as text.
"""

# %%
import warnings

import numpy as np

import fitwise as fw

rng = np.random.default_rng(11)
t = np.linspace(0, 600, 61)  # s
temperature = 21.5 + 58 * np.exp(-t / 180) + rng.normal(0, 0.2, t.size)  # °C
temperature[[9, 26, 44]] += [4.0, -3.5, 5.0]  # corrupted readings

model = "T_room + dT*exp(-x/tau)"

# %%
# The ordinary fit warns that χ² is far too large for the given uncertainty of 0.2 °C:
with warnings.catch_warnings(record=True):
    ordinary = fw.fit(model, t, temperature, yerr=0.2)
print(ordinary)

# %%
# The robust fit identifies the three readings and is no longer pulled by them:
robust = fw.fit(model, t, temperature, yerr=0.2, loss="huber")
print(robust)
print("downweighted readings at t =", t[robust.downweighted], "s")

# %%
fig, axes = robust.plot()
axes[0].set_ylabel("T (°C)")
axes[1].set_xlabel("t (s)")

# %%
# Cooling time constants in comparison (true value 180 s):
for label, r in [("ordinary", ordinary), ("robust", robust)]:
    tau = r.params["tau"]
    print(f"{label:8s} tau = {tau.value:.1f} ± {tau.stderr:.1f} s")
