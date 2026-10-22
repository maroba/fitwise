# Models

## Built-in models

| Model | Parameters | Use |
|---|---|---|
| `constant` | `c` | offset |
| `linear` | `slope`, `intercept` | lines, backgrounds |
| `polynomial(n)` | `c0` … `cn` | smooth backgrounds |
| `gaussian` | `amp`, `cen`, `sigma` | peaks; `amp` is the height |
| `lorentzian` | `amp`, `cen`, `gamma` | resonances; `gamma` is the half width |
| `voigt` | `amp`, `cen`, `sigma`, `gamma` | spectral lines |
| `exponential` | `amp`, `tau` | decays, `amp · exp(−x/τ)` |
| `sigmoid` | `amp`, `cen`, `width` | steps, titrations |
| `powerlaw` | `amp`, `exponent` | scaling laws |
| `damped_sine` | `amp`, `freq`, `phase`, `tau` | oscillations |

Shapes such as `exponential` and `sigmoid` have no offset of their own. Add one with
`+ fw.models.constant`, see below.

Every built-in model can be evaluated directly, which is handy for simulations:

```python
fw.models.gaussian(x, amp=2.0, cen=0.0, sigma=1.0)
```

## Your own functions

Any function whose first argument is the independent variable works:

```python
def arrhenius(T, A=1e10, Ea=50e3):
    return A * np.exp(-Ea / (8.314 * T))

result = fw.fit(arrhenius, T, k, yerr=dk)
```

Parameter names come from the signature. Numeric defaults are used as starting values; they
only need the right order of magnitude, because fitwise normalizes all parameters internally.
For more control, wrap the function in {class}`fitwise.Model` and give it a `guess` function.

`x` may also be two-dimensional, shape `(n, k)`, for surface fits; it is passed to your
function unchanged.

## Composition

Models combine with `+` and `*`:

```python
model = fw.models.gaussian + fw.models.linear
model = fw.models.exponential + fw.models.constant
```

If a name occurs twice, prefixes are added automatically:

```python
(gaussian + gaussian).param_names
# ('g1_amp', 'g1_cen', 'g1_sigma', 'g2_amp', 'g2_cen', 'g2_sigma')
```

Choose your own prefixes with `gaussian.with_prefix("peak_")`.

## Starting values

For each parameter, the first applicable rule wins, and the report shows which one:

1. `p0` given in `fit()`
2. a default value in the function signature
3. the heuristic of a built-in model (peak position and width, log-linear fit of a decay,
   periodogram of an oscillation, …)
4. a fallback: the middle of the bounds, or 1

Sums of built-in models are estimated step by step: first the background from the flat edges
of the data, then each peak or decay on what remains.

All built-in models and the common combinations converge without any starting values in the
test suite (100 random problems each). For your own functions without defaults, give `p0` or
defaults of the right magnitude. Automatic starting values for arbitrary functions are planned
for version 0.2.
