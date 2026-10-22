"""
Two peaks on a sloped background
================================

A spectrum with two overlapping emission lines on a linear background. Models are combined
with ``+``. Parameter names that occur twice get automatic prefixes; to choose your own
names, use :meth:`~fitwise.Model.with_prefix`.

The wavelength is measured relative to the center of the spectrum (655 nm). The intercept of
the background is then the background *in* the spectrum. With absolute wavelengths it would be
the background at 0 nm, an extrapolation over 650 nm, and slope and intercept would be almost
fully correlated; fitwise would warn about that.
"""

# %%
import numpy as np

import fitwise as fw
from fitwise.models import gaussian, linear

rng = np.random.default_rng(10)
wavelength = np.linspace(640, 670, 300)  # nm
intensity = (
    gaussian(wavelength, 820, 651.2, 1.1)
    + gaussian(wavelength, 310, 655.0, 1.4)
    + 2.5 * (wavelength - 640)
    + 40
)
intensity = rng.poisson(intensity).astype(float)
shift = wavelength - 655.0  # nm relative to the center

model = gaussian.with_prefix("a_") + gaussian.with_prefix("b_") + linear
print(model.param_names)

# %%
# The starting values come from a sequential estimate: background from the edges, then the
# strongest peak, then the second peak on what remains.
result = fw.fit(model, shift, intensity, yerr=np.sqrt(intensity))
print(result)

# %%
fig, axes = result.plot()
x = np.linspace(-15, 15, 600)
p = result.params.values_dict()
background = linear.eval(x, p)
for prefix, color in [("a_", "C2"), ("b_", "C3")]:
    peak = gaussian.with_prefix(prefix).eval(x, p)
    axes[0].fill_between(x, background, background + peak, color=color, alpha=0.3, lw=0)
axes[0].set_ylabel("counts")
axes[1].set_xlabel("wavelength − 655 nm")

# %%
# Line positions in absolute wavelength:
for prefix in ("a_", "b_"):
    cen = result.params[prefix + "cen"]
    print(f"line {prefix[0]}: {655 + cen.value:.3f} ± {cen.stderr:.3f} nm")

# %%
# The line ratio with its uncertainty. ``as_ufloat()`` takes the correlation of the two
# amplitudes into account automatically:
u = result.params.as_ufloat()
ratio = u["b_amp"] / u["a_amp"]
print(f"I_b / I_a = {ratio:.2uS}")
