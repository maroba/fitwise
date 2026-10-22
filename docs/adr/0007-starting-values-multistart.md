# 7. Starting values for user functions: multi-start with variable projection

*Status: accepted · 2026-10-22*

**Context.** Built-in models have heuristics for their starting values. Plain Python functions
have none: without `p0` or default values, every parameter started at 1. The plan proposed a
*role analysis*, which perturbs each parameter, classifies it as offset, amplitude, shift or
scale, and estimates it from the data, with multi-start only as a fallback.

To decide, we built a benchmark: 32 realistic models from lab practice as plain functions
without defaults (`tests/free_models.py`), each with random true parameters over typical
ranges. These include awkward scales such as time stamps up to 1000 s, wavelengths around
500 nm, Arrhenius with Eₐ ≈ 5·10⁴ and Shockley with Iₛ ≈ 10⁻¹². Success means: converged,
uncertainties available, and χ² plausible (p > 10⁻⁶).

**Decision.** If some free parameters have no starting value and the model is not linear,
`fit()` uses a multi-start:

1. Parameters in which the model is jointly linear, with the others held fixed, are found
   numerically at data-adapted test points. Test points where a parameter has no effect at all
   are skipped. Such linear parameters are always solved exactly (variable projection).
2. 400 candidates for the remaining parameters without starting values are drawn from a
   mixture: characteristic values of the data with jitter (position of the dominant feature,
   steepest slope, width, dominant frequency, range and spacing of x, their reciprocals),
   uniform positions within the x range, and log-uniform magnitudes of both signs over
   |log₁₀ max|x|| + 4 decades. Each candidate is scored by χ² after projection, which costs
   k + 2 evaluations.
3. The 5 best candidates are refined by the normal nonlinear solver (ADR 5), and the best result
   wins. The seed is fixed, so results are reproducible.

The role analysis is not implemented. Steps 1–3 exceed the plan's target of 85 % by a wide
margin, so the role analysis would only add complexity.

**Evidence.**

| Benchmark | Before (start at 1) | Multi-start |
|---|---|---|
| 32 models × 100 random cases | 78.9 % | **99.7 %** (3290/3300) |
| NIST StRD, no starting values, lower | – | 7/8 |
| NIST StRD, no starting values, average | – | 10/11 |
| NIST StRD, no starting values, higher | – | 8/8 |

The NIST criterion is "certified RSS reached". Several NIST problems have interchangeable
terms (Lanczos, MGH17, ENSO), so permuted parameters are an equally valid solution.

**Known limits.** Gauss2 and Gauss3 (a decay plus two strongly overlapping Gaussians, 5 nonlinear
parameters) are not solved without starting values. More candidates (5000) and refinements (20)
do not help either; blind sampling does not find this structure. Built-in models solve it,
because they guess sequentially: `exponential + gaussian + gaussian`. The plan's target of 100 %
for lower and average difficulty is therefore missed by these two problems.

**Consequences.**

- Fits of user functions without starting values take about 60 ms instead of 10 ms (benchmark
  mean). Fits with `p0`, signature defaults or built-in heuristics are unaffected.
- The report shows `multi-start` as the start strategy and `… with multi-start` as the method.
  The solver log names the projected parameters.
- The benchmark also revealed a bug in 0.1. The covariance used the Jacobian of the first
  solver run, normalized by the starting values. With starting values off by many orders of
  magnitude, this produced a spurious "singular covariance". It now uses the last run, which
  starts at the solution.
