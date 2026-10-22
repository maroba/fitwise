# Uncertainties and warnings

## What the standard errors mean

fitwise uses one documented convention (see {doc}`../adr/0003-covariance-convention`):

- **With `yerr`**: the uncertainties are taken as *absolute* one-sigma errors. The parameter
  covariance is `(JᵀWJ)⁻¹` and is **not** rescaled. χ²/ndf then tells you whether the given
  uncertainties are realistic.
- **Without `yerr`**: the noise level is estimated from the scatter of the residuals and the
  covariance is scaled by χ²/ndf. The report says so explicitly, and shows the residual standard
  deviation σ_res instead of χ²/ndf.

`scale_covar=True` or `False` overrides the default. This is the choice that
`scipy.optimize.curve_fit` calls `absolute_sigma`, and fitwise's results agree with
`curve_fit` under the same setting.

The uncertainties are calibrated. In the radioactive-decay example, repeated with 300
different random seeds, the pulls (fit − truth)/σ have a mean of −0.06 and a standard deviation
of 1.03, as they should.

## Confidence and prediction bands

```python
band = result.confidence_band(x_new, level=0.95)   # where the true curve lies
band = result.prediction_band(x_new, level=0.95)   # where a new measurement falls
band.lower, band.center, band.upper
result.plot(band="prediction")                     # default: band="confidence"
```

Both bands use the delta method, σ_f(x)² = J(x) C J(x)ᵀ, where J(x) holds the derivatives of the
model with respect to the parameters. With `yerr`, the uncertainties count as known and the
bands use the normal distribution. Without `yerr`, the scatter is estimated from the residuals,
and the bands use Student's t with ndf degrees of freedom. The prediction band adds the scatter
of a new measurement: `yerr`, or `yerr_new=` if `yerr` differs between points, or the residual
scatter without `yerr`.

In Monte Carlo tests (400 data sets each, peaks, decays and lines), 95 % bands cover the true
curve and new measurements in 93–97 % of the cases.

## Calculations with fitted parameters

`result.params.as_ufloat()` returns the parameters as correlated values of the
[uncertainties](https://uncertainties.readthedocs.io) package (`pip install "fitwise[units]"`).
Error propagation then includes the correlations automatically:

```python
u = result.params.as_ufloat()
Q = np.pi * u["freq"] * u["tau"]
print(f"{Q:.1uS}")       # e.g. 15.55(9)
```

## How often ±1σ contains the truth

With `yerr`, the interval value ± standard error contains the true value in 68.3 % of the
cases, as expected. Without `yerr`, σ itself is estimated from the residuals. With few
degrees of freedom the interval then covers less: with 10 degrees of freedom only 65.9 %
(Student's t). The bands account for this, but the plain standard error cannot. fitwise's
Monte Carlo tests check both cases.

## Reading χ²/ndf

With correct uncertainties and a correct model, χ²/ndf scatters around 1, and the p-value is
uniformly distributed between 0 and 1.

- **χ²/ndf ≫ 1** (small p): the model does not describe the data, or `yerr` is too small.
- **χ²/ndf ≪ 1** (p close to 1): `yerr` is too large, or the points are not independent.

fitwise warns if χ² falls outside the central 99 % of its distribution.

## Warnings

Each warning is stored in `result.warnings`, printed in the report and emitted through Python's
`warnings` module, so it shows up in scripts and notebooks.

| Warning | Meaning | What to do |
|---|---|---|
| `NotConvergedWarning` | The solver stopped before meeting its criteria. `result.ok` is `False`. | Better starting values (`p0`), bounds, or a simpler model |
| `SingularCovarianceWarning` | Some parameter combination is not determined by the data. No uncertainties are reported. | Remove redundant parameters, e.g. `(a + b) · x` |
| `ParameterAtBoundWarning` | A parameter ended on a bound; its error bar is not meaningful. | Check whether the bound is physical |
| `PoorChiSquareWarning` | χ²/ndf is implausible for the given `yerr`. | Check the model and the uncertainties |
| `StrongCorrelationWarning` | \|r\| > 0.999: only a combination of two parameters is determined. | Use the covariance when combining them, or reparametrize |
| `LargeUncertaintyWarning` | A standard error exceeds the value. | Often the parameter is compatible with zero; consider fixing it |

Correlations between 0.5 and 0.999 are listed in the report as information. They are common and
harmless for many physical laws (Arrhenius, power laws, lines far from x = 0), as long as you
use the covariance when you compute something from several parameters.

Silence a warning type with the standard mechanism, for example:

```python
import warnings
warnings.simplefilter("ignore", fw.warnings.LargeUncertaintyWarning)
```

The warning still appears in `result.warnings` and in the report.
