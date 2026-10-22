"""
Radioactive decay with background
=================================

Counts from a short-lived isotope are recorded in 10-s intervals. The counts follow Poisson
statistics, so the uncertainty of each value is √N. A constant background from the
environment adds to the decay.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(5)
t = np.arange(0, 600, 10.0)  # s
expected = 850 * np.exp(-t / 140) + 25
counts = rng.poisson(expected).astype(float)

result = fw.fit(fw.models.exponential + fw.models.constant, t, counts, yerr=np.sqrt(counts))
print(result)

# %%
# The half-life follows from the decay time: T½ = τ · ln 2.
tau = result.params["tau"]
print(f"T½ = {tau.value * np.log(2):.1f} ± {tau.stderr * np.log(2):.1f} s")

# %%
fig, axes = result.plot()
axes[0].set_yscale("log")
axes[0].set_ylabel("counts per 10 s")
axes[1].set_xlabel("t (s)")
