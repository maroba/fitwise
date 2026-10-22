"""
Resonance curve
===============

The amplitude of a driven oscillator near resonance is approximately Lorentzian. The quality
factor follows from the resonance frequency and the width: Q = f₀ / (2γ), where γ is the half
width at half maximum. The detector has a small constant offset.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(9)
f = np.linspace(950, 1050, 101)  # Hz
amplitude = 2.4 * 4.0**2 / ((f - 1002.5) ** 2 + 4.0**2) + 0.05 + rng.normal(0, 0.02, f.size)

result = fw.fit(fw.models.lorentzian + fw.models.constant, f, amplitude, yerr=0.02)
print(result)

# %%
f0, gamma = result.params["cen"], result.params["gamma"]
Q = f0.value / (2 * gamma.value)
print(
    f"f0 = {f0.value:.2f} ± {f0.stderr:.2f} Hz,  Q = {Q:.0f} ± {Q * gamma.stderr / gamma.value:.0f}"
)

# %%
fig, axes = result.plot()
axes[0].set_ylabel("amplitude (V)")
axes[1].set_xlabel("f (Hz)")
