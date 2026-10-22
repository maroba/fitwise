# 2. Orthogonal distance regression via `odrpack`

*Status: accepted · 2026-10-22 · implemented in 0.3*

**Context.** Fits with uncertainties in x and y need orthogonal distance regression (ODR).
`scipy.odr` is deprecated since SciPy 1.17 and will be removed in SciPy 1.19. The SciPy project
recommends the separate [`odrpack`](https://pypi.org/project/odrpack/) package, which wraps
ODRPACK95 and, unlike `scipy.odr`, supports parameter bounds.

**Decision.** The ODR backend uses `odrpack` from the start and never `scipy.odr`. `odrpack` is an
optional dependency (`fitwise[odr]`). Without it, `xerr` raises an error that says which extra
to install.

**Consequences.** No migration is needed later, and bounds work the same way in every backend.
fitwise now depends on a younger project with a smaller maintainer base. If `odrpack` turns out
to be unavailable on a supported platform, the fallback is ODR implemented as an extended
residual vector solved with `scipy.optimize.least_squares`.

**Implementation notes (0.3).**

- Two stages: an ordinary fit in y provides the starting values, then ODR runs from there.
- `cov_beta` from `odrpack` is unscaled, like in `scipy.odr`, and is used directly with absolute
  uncertainties. `irank` is the rank *deficiency*. Convergence is read from the last digit of
  `info`; a leading 4 (questionable, e.g. rank deficient) is reported through the covariance
  warning.
- ODRPACK fails if the solution lies on a bound, even when started inside it, and its
  result is then wrong. The constrained solution with the parameter held fixed differs. fitwise
  reports such fits as not converged and recommends `fixed=`. The `least_squares` fallback
  mentioned above was not needed for any other case and is not implemented.
- Verified against the York/Pearson reference solution, `scipy.odr` (nonlinear model), the limit
  xerr → 0, and Monte Carlo calibration of the standard errors.
