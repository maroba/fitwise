# fitwise

**One call. One fit. One report you can trust.**

[![Tests](https://github.com/maroba/fitwise/actions/workflows/test.yml/badge.svg)](https://github.com/maroba/fitwise/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/fitwise.svg)](https://pypi.org/project/fitwise/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> **Status: alpha (0.2).** Fitting, built-in and text models, automatic starting values,
> robust fits, report, plot and export work and are tested against the NIST reference data.
> The API may still change before 1.0.
> **Documentation: https://maroba.github.io/fitwise/**

`fitwise` is an ergonomic layer over [`lmfit`](https://lmfit.github.io/lmfit-py/) and
[`scipy`](https://scipy.org/) for curve fitting in the lab. It brings no new solver of its own.
Instead it makes the existing ones fast to use and hard to misuse:

```python
import fitwise as fw

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)              # report: parameters ± errors, χ²/ndf, R², correlations, warnings
result.plot()              # data, fit, confidence band, residuals
result.params["sigma"]     # value, uncertainty, unit
```

## What it does for you

- **Picks the solver:** direct linear least squares when the model is linear in its parameters,
  Levenberg–Marquardt otherwise, and orthogonal distance regression when `xerr` is given.
- **Finds starting values,** including for your own functions and for models written as
  text (`fw.fit("a*exp(-x/tau) + c", x, y)`), and shows how it found them.
- **Resists outliers** on request: `loss="huber"` downweights them and marks them in the plot.
- **Gives honest uncertainties:** a documented covariance convention, confidence and prediction
  bands, and bootstrap or profile likelihood on request.
- **Warns instead of returning wrong numbers:** non-convergence, parameters at a bound, poor
  χ²/ndf and strong correlations all show up in the report.
- **Makes fits reproducible:** `result.to_dict()` captures everything needed to repeat the fit.

## Installation

```bash
pip install fitwise            # core: numpy, scipy, lmfit
pip install "fitwise[all]"     # + plotting, ODR, units, symbolic models, batch fits
```

## Development

```bash
uv sync --all-extras           # create .venv with all extras and dev tools
uv run pre-commit install
uv run pytest                  # fast tests
uv run pytest -m nist          # NIST StRD reference suite
```

## License

BSD-3-Clause
