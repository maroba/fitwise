# 5. Nonlinear solver: `scipy.optimize.least_squares` with normalization and restarts

*Status: accepted · 2026-10-22*

**Context.** The v0.1 goal is that all 27 NIST StRD nonlinear problems reach the certified values
(≥ 6 significant digits) from both official starting points. The original plan wrapped
`lmfit.minimize`. Measured with tight tolerances:

| Setup | Start 1 | Start 2 |
|---|---|---|
| `lmfit` `leastsq`, default tolerances | 17/27 | 16/27 |
| `lmfit` `least_squares` (TRF), tol 1e-12 | 21/27 | 22/27 |
| scipy `least_squares` LM, 2-point Jacobian | 20/27 | 23/27 |
| scipy TRF, 3-point Jacobian, normalized parameters, restarts | **27/27** | **27/27** |

Three things made the difference:

1. **Normalized parameters** `u = p / |p_start|`. Without them Hahn1 (coefficients between 1 and
   1e-7) stops early at LRE 2.3. scipy's `x_scale` option does not have the same effect.
2. **Central (3-point) finite differences** for the Jacobian. Kirby2 and Lanczos3 need them.
3. **Restarts** from the solution with fresh normalization, until χ² stops improving. They fix
   MGH09, MGH17, BoxBOD and Bennett5 from start 1.

Wrapping lmfit's minimizer leaves no room for the first and third measures.

**Decision.** The default backend calls `scipy.optimize.least_squares` directly, with Trust Region
Reflective, 3-point finite differences, normalized parameters and up to 10 restarts. TRF also
handles bounds, so bounded and unbounded fits use the same solver. Levenberg–Marquardt stays
available as `method="lm"`. For LM, fitwise computes the central differences itself, because
older scipy versions (for example 1.12) silently fall back to 2-point for `lm`. lmfit stays a dependency for profile
likelihood (`conf_interval`, v0.3) and expression models (`asteval`, v0.2).

The covariance is `(JᵀJ)⁻¹` from an SVD of the final Jacobian in normalized coordinates. The
Jacobian counts as rank deficient if `s_min / s_max ≤ 1e-9`. Redundant parameters reach about
1e-12 there, and the worst well-posed NIST problem (Nelson) 3e-6.

**Consequences.**

- NIST StRD 27/27 from both starting points, about 0.25 s for all 27 problems.
- Results agree with `curve_fit` to better than 1e-5 standard errors, and the covariance agrees
  to 1e-4 relative (different finite-difference schemes).
- `FitResult.raw` is a `scipy.optimize.OptimizeResult`, not an lmfit object.
- Restarts cost extra model evaluations (typically 2–3 runs), which is negligible for curve
  fitting.
