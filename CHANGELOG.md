# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Linear models are detected automatically and solved directly, without starting values;
  bounds use `scipy.optimize.lsq_linear`. `method="linear"` requests this explicitly.
- NIST StRD linear regression suite (11 datasets) in the tests.
- User functions without starting values: multi-start with variable projection (99.7 % on a
  benchmark of 32 lab models). `method="multistart"` forces it for all parameters.
- Benchmark catalog of 32 realistic models in the tests.
- Models as text: `fw.fit("a*exp(-x/tau) + c", x, y)`, validated against an allowlist;
  `^` means power. `fw.ExpressionModel` for other variable names.
- Robust fits against outliers: `loss="soft_l1" | "huber" | "cauchy" | "arctan"` with
  `loss_scale`; downweighted points are reported, marked in the plot and available as
  `result.downweighted`.

### Changed

- `method="auto"` reports why it chose a solver: `auto: linear in the parameters` or
  `auto: nonlinear`.

## [0.1.1] - 2026-10-22

### Fixed

- Uncertainties were reported as unavailable ("singular covariance") when the starting values
  were off by many orders of magnitude, although the fit itself had converged. The covariance
  now uses the Jacobian of the last solver run, which starts at the solution, instead of the
  first run's Jacobian normalized by the starting values.

## [0.1.0] - 2026-10-22

First public release.

### Added

- `fitwise.fit()` with `yerr`, `p0`, `bounds`, `fixed`, `scale_covar` and `nan_policy`.
- `Model` from plain Python functions; composition with `+` and `*`, automatic prefixes.
- Built-in models: `constant`, `linear`, `polynomial`, `gaussian`, `lorentzian`, `voigt`,
  `exponential`, `sigmoid`, `powerlaw`, `damped_sine`, all with starting-value heuristics.
- Nonlinear least squares on `scipy.optimize.least_squares` with normalized parameters and
  restarts; all 27 NIST StRD problems are solved from both starting points.
- `FitResult` with goodness-of-fit statistics, a documented covariance convention and
  warnings for questionable results.
- Text report, `plot()` with residual panel, `to_dict()` for JSON export.
- Documentation: quickstart, guides on models and uncertainties, 10 gallery examples.

[Unreleased]: https://github.com/maroba/fitwise/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/maroba/fitwise/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maroba/fitwise/releases/tag/v0.1.0
