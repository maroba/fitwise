# 2. Orthogonal distance regression via `odrpack`

*Status: accepted · 2026-10-22*

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
