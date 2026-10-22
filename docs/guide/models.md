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

## Models as text

For quick fits, write the model as an expression:

```python
result = fw.fit("a*exp(-x/tau) + c", x, y, yerr=dy)
```

Parameters are all names except `x`, the functions and `pi`, in order of first appearance.
Available functions include `exp`, `log`/`ln`, `log10`, `sqrt`, `sin`, `cos`, `tan` and their
inverses, the hyperbolic functions, `abs`, `erf`, `erfc` and `heaviside`; see
{data}`fitwise.expression.FUNCTIONS`. As in Origin, Excel and MATLAB, `^` means power. For
another variable name, use `fw.ExpressionModel("a*exp(-t/tau)", independent="t")`.

Expressions are checked against an allowlist before anything runs. Only numbers, names,
`+ - * / ** ^` and these functions are accepted, so an expression cannot access files or
attributes.

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

## Linear models need no starting values

If a model is linear in its parameters, such as a line, a polynomial, or `a·sin(x) + b·x²`,
fitwise recognizes this automatically and solves the problem directly, without iterations or
starting values. The report then shows `Linear least squares (direct)` and `not needed (linear
model)`. This also works for your own functions, and it reaches the certified values of the
NIST linear reference datasets, including the notoriously ill-conditioned Filip polynomial.

## Starting values

For each parameter, the first applicable rule wins, and the report shows which one:

1. `p0` given in `fit()`
2. a default value in the function signature
3. the heuristic of a built-in model (peak position and width, log-linear fit of a decay,
   periodogram of an oscillation, …)
4. a multi-start for everything that is still missing (see below)

Sums of built-in models are estimated step by step: first the background from the flat edges
of the data, then each peak or decay on what remains.

### Your own functions without starting values

You do not have to give starting values for your own functions. fitwise then finds them like
this:

- Parameters the model is linear in, such as amplitudes and offsets, are always computed
  exactly.
- For the others, 400 candidates are drawn from ranges adapted to the data: positions and
  widths of features, frequencies, and many orders of magnitude of both signs. The best 5 are
  refined.

The report shows `multi-start` for these parameters.

On a benchmark of 32 realistic lab models (decays, peaks, sigmoids, Arrhenius, Shockley diode,
Planck, Hill, Morse, …) with random parameters, this reaches the global minimum in 99.7 % of
3300 fits. It costs about 60 ms per fit instead of 10 ms.

Where it can fail: several strongly overlapping features in one function, with many nonlinear
parameters, for example two overlapping peaks on a decay. Build such models from built-in
components (`fw.models.exponential + fw.models.gaussian + fw.models.gaussian`), which are guessed
one by one, or give `p0` for the positions.
