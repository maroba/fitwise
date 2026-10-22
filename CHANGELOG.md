# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/maroba/fitwise/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/maroba/fitwise/releases/tag/v0.1.0
