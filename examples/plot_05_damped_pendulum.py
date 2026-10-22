"""
Damped pendulum
===============

The angle of a pendulum is recorded with a rotary encoder. The oscillation decays
exponentially; the encoder has a small zero offset. Frequency, phase and decay time are
estimated automatically from a periodogram.
"""

# %%
import numpy as np

import fitwise as fw

rng = np.random.default_rng(5)
t = np.linspace(0, 20, 400)  # s
angle = 12 * np.exp(-t / 8) * np.sin(2 * np.pi * 0.62 * t + 0.4) + 0.3 + rng.normal(0, 0.2, t.size)

result = fw.fit(fw.models.damped_sine + fw.models.constant, t, angle, yerr=0.2)
print(result)

# %%
fig, axes = result.plot()
axes[0].set_ylabel("angle (°)")
axes[1].set_xlabel("t (s)")

# %%
# The quality factor of the pendulum is Q = π f τ. ``as_ufloat()`` returns the parameters as
# correlated values of the `uncertainties <https://uncertainties.readthedocs.io>`_ package, so
# the error propagation automatically includes the correlation between f and τ:
u = result.params.as_ufloat()
Q = np.pi * u["freq"] * u["tau"]
print(f"Q = {Q:.1uS}")

# %%
# The 95 % prediction band shows where new readings of the encoder are expected:
fig, axes = result.plot(band="prediction", residuals=False)
axes[0].set_xlim(0, 5)
axes[0].set_ylabel("angle (°)")
axes[0].set_xlabel("t (s)")
