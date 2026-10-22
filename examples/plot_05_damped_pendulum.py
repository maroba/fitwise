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
# The quality factor of the pendulum is Q = π f τ.
f, tau = result.params["freq"], result.params["tau"]
q = np.pi * f.value * tau.value
q_err = q * np.hypot(f.stderr / f.value, tau.stderr / tau.value)
print(f"Q = {q:.1f} ± {q_err:.1f}")

# %%
# The error propagation above treats f and τ as independent. Their correlation is small, so
# this is justified here. In general, use the full ``result.covariance``.
print(f"corr(freq, tau) = {result.correlations()[('freq', 'tau')]:+.3f}")
