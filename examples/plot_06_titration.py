"""
Titration curve
===============

The pH during the titration of a weak acid with sodium hydroxide rises in a step around the
equivalence point. A logistic step on top of a constant describes the curve well enough to
locate the equivalence point precisely.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(6)
volume = np.linspace(0, 20, 41)  # mL
ph = 4.0 + 6.5 / (1 + np.exp(-(volume - 12.3) / 0.45)) + rng.normal(0, 0.05, volume.size)

result = fw.fit(fw.models.sigmoid + fw.models.constant, volume, ph, yerr=0.05)
print(result)

# %%
# The equivalence point is the midpoint of the step:
cen = result.params["cen"]
print(f"V_eq = {cen.value:.3f} ± {cen.stderr:.3f} mL")

# %%
fig, axes = result.plot()
axes[0].set_ylabel("pH")
axes[1].set_xlabel("NaOH added (mL)")
