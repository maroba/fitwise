# fitwise

**One call. One fit. One report you can trust.**

:::{warning}
fitwise is in **pre-alpha**. The API may still change before v0.1.
:::

`fitwise` is an ergonomic layer over [lmfit](https://lmfit.github.io/lmfit-py/) and
[SciPy](https://scipy.org/) for curve fitting in the lab.

```python
import fitwise as fw

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)
result.plot()
```

## Where to go next

- {doc}`guide/quickstart`: installation, a first fit, reading the report
- {doc}`guide/models`: built-in models, your own functions, composition, starting values
- {doc}`guide/uncertainties`: what the error bars mean, χ²/ndf, warnings
- {doc}`auto_examples/index`: complete fits from typical lab experiments

```{toctree}
:maxdepth: 1
:hidden:

guide/quickstart
guide/models
guide/uncertainties
auto_examples/index
api
adr/index
```
