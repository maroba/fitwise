# 3. Covariance convention

*Status: accepted · 2026-10-22*

**Context.** Whether the parameter covariance is scaled by χ²/ndf (`absolute_sigma` in
`curve_fit`, `scale_covar` in lmfit) changes every reported uncertainty. The library defaults
differ, and most users are not aware of the choice.

**Decision.**

- With `yerr`, the errors are absolute 1σ uncertainties and the covariance is **not** scaled.
  χ²/ndf is reported as a consistency check, with a warning if it is implausible.
- Without `yerr`, the noise level is estimated from the residuals and the covariance **is**
  scaled. The report says so explicitly and shows the RMSE instead of χ²/ndf.
- `scale_covar=True/False` overrides the default. The choice is stored in the result and shown
  in the report.

**Consequences.** Users who supply measured uncertainties get the physically expected
behaviour. Results can differ from `lmfit`'s default (which always scales) when `yerr` is given.
The documentation explains this difference, and the reference tests cover both modes.
