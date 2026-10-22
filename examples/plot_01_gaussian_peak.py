"""
Gaussian peak in three lines
============================

The shortest possible fit: a built-in model, data with uncertainties, a report and a plot.
No starting values are needed; fitwise estimates them from the data.
"""

# %%
# Simulated measurement: a Gaussian peak with known noise of 0.05.
import numpy as np

rng = np.random.default_rng(1)
x = np.linspace(-4, 6, 80)
y = 3.2 * np.exp(-((x - 1.1) ** 2) / (2 * 0.8**2)) + rng.normal(0, 0.05, x.size)
dy = np.full_like(x, 0.05)

# %%
# The fit itself:
import fitwise as fw

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)

# %%
# ``plot()`` shows the data, the fit and the residuals in units of ``yerr``.
fig, axes = result.plot()

# %%
# Individual parameters are available by name, with value and standard error:
sigma = result.params["sigma"]
print(f"FWHM = {2.3548 * sigma.value:.3f} ± {2.3548 * sigma.stderr:.3f}")
