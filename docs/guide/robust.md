# Outliers and robust fits

Least squares gives every point full weight, so a single outlier can pull the fit away from
the bulk of the data. A robust loss downweights points that lie far from the curve:

```python
result = fw.fit(model, x, y, yerr=dy, loss="huber")
print(result)
result.plot()          # downweighted points are circled
result.downweighted    # boolean mask of the outliers
```

## How it works

1. First an ordinary fit runs, with the usual automatic choices (direct solution, starting
   values, multi-start).
2. From that solution, Trust Region Reflective minimizes the robust loss. Residuals larger
   than `loss_scale` are downweighted.

| `loss` | Behaviour |
|---|---|
| `"soft_l1"`, `"huber"` | mild: outliers count linearly instead of quadratically |
| `"cauchy"`, `"arctan"` | strong: far outliers have almost no influence |

**Scale.** With `yerr`, `loss_scale` is measured in units of `yerr` (default 1.5). Without
`yerr`, the default is 1.5 times a robust estimate of the scatter (1.4826 · MAD) of the
ordinary fit. Set `loss_scale` explicitly if you know at what deviation a point becomes
suspicious.

## Uncertainties

The covariance is that of the equivalent weighted least-squares problem, with the robust
weights at the solution. Without `yerr`, it is scaled by the robust scatter estimate, so the
outliers do not inflate it. The report shows this robust estimate as `χ²/ndf (robust)` with
`yerr`, or as `σ_res (robust)` without. With correct `yerr` it is close to 1. The χ² test and its
warning are skipped, because outliers are expected by design.

Monte Carlo check (straight line, 60 points, 5 % outliers of 10–30 σ, 300 data sets): the pulls
(fit − truth)/σ have a mean within ±0.1. Their spread is 0.95–1.09 with `yerr` for all four
losses, and 0.96–1.17 without `yerr`. The somewhat larger spread without `yerr` comes from the
statistical noise of the scatter estimate itself.

```{note}
A robust fit is no substitute for looking at the data. If many points are downweighted, the
model may be wrong, not the data.
```
