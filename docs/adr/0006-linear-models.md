# 6. Direct solution of linear models

*Status: accepted · 2026-10-22*

**Context.** Many lab fits are linear in their parameters: lines, polynomials, sums of known
functions. For them, least squares has a unique closed-form solution. An iterative solver
needs starting values here for no reason, and it can fail where no failure is necessary.

**Decision.**

- **Detection.** A model counts as linear if it is declared so (built-in models) or if the
  numerical check `f(p) = f(0) + A p` holds at two random parameter points (relative tolerance
  1e-8). The design matrix `A` comes from `k + 1` model evaluations, so arbitrary Python
  functions work; affine models with a constant term are included.
- **Solution.** The weighted design matrix is column-normalized and solved with SVD
  (`numpy.linalg.lstsq`), followed by one step of iterative refinement. With bounds,
  `scipy.optimize.lsq_linear` (BVLS) is used instead. The covariance comes from the same
  normalized matrix; the rank threshold is `s_min / s_max ≤ 1e-13`.
- `method="auto"` uses this path whenever the model is linear. `method="linear"` requests it
  explicitly and fails for nonlinear models. `method="trf"` still forces the iterative solver.

**Evidence.** We measured this on the NIST StRD linear suite (11 datasets), counting correct
significant digits of the parameters:

| Dataset | Digits | Note |
|---|---|---|
| Norris, Pontius, NoInt1, NoInt2 | 13.4–15 | |
| Longley (6 predictors) | 11.0 | |
| Wampler1–3 | 9.6–12.9 | |
| Filip (degree-10 polynomial) | 7.4 | about the limit of double precision |
| Wampler4 | 7.7 | |
| Wampler5 | 5.7 | Its certified standard errors are 10³–10⁷ times the values; the deviation is about 1e-9 standard errors. |

Standard errors match the certified ones to at least 8 digits. Without normalization, plain
SVD loses Filip completely (0 digits). Before this change, the iterative path reached 4.4
digits on Filip and 5.7 on Longley.

**Consequences.** Linear fits need no starting values, cost `k + 3` model evaluations, and are
as accurate as dedicated linear regression software. The report says `not needed (linear
model)` in the `Start` line. Results for linear models change slightly compared with 0.1, by
far less than one standard error.
