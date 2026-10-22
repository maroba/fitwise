# Units

Pass [pint](https://pint.readthedocs.io) quantities, and the parameters come out with units
(`pip install "fitwise[units]"`):

```python
import pint
import fitwise as fw

u = pint.UnitRegistry()
result = fw.fit(fw.models.linear, current * u.mA, voltage * u.V, yerr=50 * u.mV)

result.params["slope"].unit                      # V/mA
result.params["slope"].measurement.to("ohm")     # (468.4 ± 2.6) Ω
result.eval([0.01] * u.A)                        # quantity in V
```

The report gets a unit column, and the plot labels its axes with the units.

## Rules

- `yerr` and `xerr` are converted to the units of `y` and `x`. Plain numbers are taken to be in
  those units already.
- `p0`, `bounds` and `fixed` accept quantities in any compatible unit, for example
  `p0={"tau": 0.01 * u.s}` for data in milliseconds.
- `result.eval()`, the bands and `to_dict()` understand quantities too.

## Where the parameter units come from

**Built-in models** declare how each parameter scales with the units of x and y:

| Parameter | Unit |
|---|---|
| `amp`, `c`, `intercept`, Gaussian/Lorentzian heights | y |
| `cen`, `sigma`, `gamma`, `tau`, `width` | x |
| `slope` | y/x |
| polynomial `c_i` | y/xⁱ |
| `freq` | 1/x |
| `phase`, power-law `exponent` | dimensionless |
| power-law `amp` | not declared (y/x^exponent depends on the fit) |

Sums of models keep the units of their parts. Products mix units, so their parameters get none.

**Your own functions and expressions** state the units they need:

```python
fw.fit("a*exp(-x/tau)", t, v, units={"tau": "ms", "a": "V"})
```

fitwise does not try to infer units of arbitrary functions automatically. That would mean
trying out unit combinations, which is slow with pint and often ambiguous.
