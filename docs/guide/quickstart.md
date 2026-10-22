# Quickstart

## Installation

```bash
pip install "fitwise[plot]"
```

The core needs only NumPy, SciPy and lmfit; `[plot]` adds matplotlib.

## A first fit

```python
import numpy as np
import fitwise as fw

x = np.linspace(-4, 6, 80)
y = ...            # your measured values
dy = 0.05          # their one-sigma uncertainty (scalar or array)

result = fw.fit(fw.models.gaussian, x, y, yerr=dy)
print(result)
result.plot()
```

`print(result)` gives a report like this:

```text
Model:      gaussian(x; amp, cen, sigma)
Method:     Trust Region Reflective (scipy least_squares) (auto)
Converged:  yes (12 evals)
Data:       80 points, yerr given (absolute)
Start:      amp, cen, sigma: peak heuristic

Parameter    Value   ± Std.Err    Rel.   Bounds
amp          3.198       0.020   0.6 %   –
cen         1.0948      0.0062   0.6 %   –
sigma       0.8058      0.0057   0.7 %   –

χ²/ndf = 0.718 (77 dof, p = 0.97)    R² = 0.9983    AIC = -23.6
Warnings:   none
```

Every line has a purpose:

- **Method** and **Start** say which solver ran and where the starting values came from.
  Nothing is decided silently.
- **Value ± Std.Err** are rounded consistently: the uncertainty to two significant digits, the
  value to the same decimal place.
- **χ²/ndf** and its p-value test whether model and uncertainties fit together
  (see {doc}`uncertainties`).
- **Warnings** lists everything that should make you look twice. An empty list is part of the
  result.

## Working with the result

```python
p = result.params["sigma"]      # or result.params.sigma
p.value, p.stderr               # best fit and one-sigma error
result.params.values_dict()     # {"amp": ..., "cen": ..., "sigma": ...}
result.eval(x_new)              # evaluate the fitted model anywhere
result.covariance               # covariance matrix of the free parameters
result.correlations(0.5)        # strongly correlated pairs
result.ok                       # converged and uncertainties available
result.warnings                 # tuple of FitWarning objects
result.to_dict()                # JSON-serializable record, e.g. for a lab journal
```

## Controlling the fit

```python
fw.fit(model, x, y, yerr=dy,
       p0={"sigma": 0.5},              # starting values (any subset)
       bounds={"sigma": (0, None)},    # None = open side
       fixed={"cen": 1.1})             # held constant
```

Next: {doc}`models` explains built-in models, your own functions and composition;
the {doc}`../auto_examples/index` show complete fits from typical lab experiments.
