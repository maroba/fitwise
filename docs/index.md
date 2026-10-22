# fitwise

**One call. One fit. One report you can trust.**

:::{warning}
fitwise is in **pre-alpha**. The API shown here is the design target for v0.1 and is not
implemented yet.
:::

`fitwise` is an ergonomic layer over [lmfit](https://lmfit.github.io/lmfit-py/) and
[SciPy](https://scipy.org/) for curve fitting in the lab.

```python
import fitwise as fw

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)
result.plot()
```

```{toctree}
:maxdepth: 1
:hidden:

auto_examples/index
api
adr/index
```
