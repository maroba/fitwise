"""Models: a function of ``x`` and named parameters, plus metadata for guessing and reporting."""

from __future__ import annotations

import copy
import inspect
import keyword
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fitwise._typing import ArrayLike, FloatArray

__all__ = ["CompositeModel", "Model", "as_model"]

Shape = Literal["peak", "monotone", "oscillation", "baseline", "other"]
GuessFunc = Callable[[FloatArray, FloatArray], Mapping[str, float]]

_OPS: dict[str, Callable[[FloatArray, FloatArray], FloatArray]] = {
    "+": np.add,
    "*": np.multiply,
}


class Model:
    """A fit model built from a Python function.

    Parameters
    ----------
    func : callable
        ``func(x, p1, p2, ...)`` returning an array shaped like ``y``. The first argument is the
        independent variable; all further positional arguments are fit parameters. Numeric
        default values are used as starting values.
    name : str, optional
        Name shown in the report. Defaults to ``func.__name__``.
    param_names : sequence of str, optional
        Parameter names, required if they cannot be read from the signature (``func(x, *p)``).
    guess : callable, optional
        ``guess(x, y) -> {name: value}`` with starting values estimated from the data.
    guess_label : str, optional
        Short description of the guess heuristic for the report.
    linear : bool, optional
        Whether the model is linear in its parameters; ``None`` if unknown.
    shape : {"peak", "monotone", "oscillation", "baseline", "other"}
        Qualitative shape; used to guess starting values of composite models.
    symmetric : sequence of str
        Parameters the model depends on only through their absolute value, such as a
        Gaussian width. Their sign is normalized to positive after the fit.
    abbrev : str, optional
        Short name for automatic prefixes in composite models (``g1_amp``).
    formula : str, optional
        The model equation in LaTeX, for documentation and export.
    param_docs : mapping, optional
        ``{name: description}`` for documentation.

    Examples
    --------
    >>> def decay(t, amp, tau=1.0):
    ...     return amp * np.exp(-t / tau)
    >>> model = Model(decay)
    >>> model.param_names
    ('amp', 'tau')
    >>> model + Model(lambda t, c: c + 0 * t)
    CompositeModel(decay(x; amp, tau) + model(x; c))
    """

    def __init__(
        self,
        func: Callable[..., ArrayLike],
        *,
        name: str | None = None,
        param_names: Sequence[str] | None = None,
        guess: GuessFunc | None = None,
        guess_label: str | None = None,
        linear: bool | None = None,
        shape: Shape = "other",
        symmetric: Sequence[str] = (),
        abbrev: str | None = None,
        formula: str | None = None,
        param_docs: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError(f"model function must be callable, got {func!r}")
        self.func = func
        default_name = getattr(func, "__name__", "model")
        self.name = name or ("model" if default_name == "<lambda>" else default_name)
        self._base_names, self._base_defaults = _introspect(func, param_names)
        unknown = set(symmetric) - set(self._base_names)
        if unknown:
            raise ValueError(f"symmetric names {sorted(unknown)} are not parameters")
        self._guess = guess
        self.guess_label = guess_label or f"{self.name} heuristic"
        self.linear = linear
        self.shape: Shape = shape
        self.symmetric = frozenset(symmetric)
        self.abbrev = abbrev or self.name[:1].lower()
        self.formula = formula
        self.param_docs = dict(param_docs or {})
        self.prefix = ""
        self._auto_prefix = False

    # --- names --------------------------------------------------------------------------

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names including the prefix."""
        return tuple(self.prefix + n for n in self._base_names)

    @property
    def defaults(self) -> dict[str, float]:
        """Starting values from the function signature."""
        return {self.prefix + n: v for n, v in self._base_defaults.items()}

    @property
    def leaves(self) -> tuple[Model, ...]:
        """The elementary models this model is composed of."""
        return (self,)

    def with_prefix(self, prefix: str) -> Model:
        """Return a copy whose parameter names start with ``prefix``.

        Examples
        --------
        >>> from fitwise.models import gaussian
        >>> gaussian.with_prefix("peak_").param_names
        ('peak_amp', 'peak_cen', 'peak_sigma')
        """
        if prefix and not (prefix + "x").isidentifier():
            raise ValueError(f"prefix {prefix!r} must start a valid identifier")
        return self._prefixed(prefix, auto=False)

    def _prefixed(self, prefix: str, *, auto: bool) -> Model:
        new = copy.copy(self)
        new.prefix = prefix
        new._auto_prefix = auto
        return new

    def _without_auto_prefix(self) -> Model:
        return self._prefixed("", auto=False) if self._auto_prefix else self

    def _replace_leaves(self, it: Iterator[Model]) -> Model:
        return next(it)

    # --- evaluation ---------------------------------------------------------------------

    def eval(self, x: ArrayLike, params: Mapping[str, float]) -> FloatArray:
        """Evaluate the model at ``x`` for the parameter values in ``params``."""
        args = [params[self.prefix + n] for n in self._base_names]
        return np.asarray(self.func(x, *args), dtype=float)

    def __call__(self, x: ArrayLike, *args: float, **kwargs: float) -> FloatArray:
        """Evaluate with positional or keyword parameters, e.g. ``gaussian(x, 1, 0, 2)``."""
        return self.eval(x, _bind(self, args, kwargs))

    # --- guessing and normalization -----------------------------------------------------

    def guess(self, x: ArrayLike, y: ArrayLike) -> dict[str, float]:
        """Estimate starting values from data; may return only some parameters."""
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return {n: v for n, (v, _) in self._guess_labeled(xa, ya).items()}

    def _guess_labeled(self, x: FloatArray, y: FloatArray) -> dict[str, tuple[float, str]]:
        if self._guess is None:
            return {}
        try:
            raw = self._guess(x, y)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            return {}
        return {
            self.prefix + n: (float(v), self.guess_label)
            for n, v in raw.items()
            if n in self._base_names and math.isfinite(v)
        }

    def canonicalize(
        self, values: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Map equivalent solutions to a canonical one.

        Returns the new values and the sign (``±1``) applied to each parameter, so that the
        covariance can be transformed consistently.
        """
        new = dict(values)
        signs = dict.fromkeys(values, 1.0)
        for leaf in self.leaves:
            for base in leaf.symmetric:
                name = leaf.prefix + base
                if name in new and new[name] < 0:
                    new[name] = -new[name]
                    signs[name] = -1.0
        return new, signs

    # --- composition and display --------------------------------------------------------

    def __add__(self, other: Model | Callable[..., ArrayLike]) -> CompositeModel:
        return CompositeModel(self, as_model(other), "+")

    def __radd__(self, other: Callable[..., ArrayLike]) -> CompositeModel:
        return CompositeModel(as_model(other), self, "+")

    def __mul__(self, other: Model | Callable[..., ArrayLike]) -> CompositeModel:
        return CompositeModel(self, as_model(other), "*")

    def __rmul__(self, other: Callable[..., ArrayLike]) -> CompositeModel:
        return CompositeModel(as_model(other), self, "*")

    def signature(self) -> str:
        """Human-readable signature, e.g. ``gaussian(x; amp, cen, sigma)``."""
        return f"{self.name}(x; {', '.join(self.param_names)})"

    def __repr__(self) -> str:
        return f"Model({self.signature()})"


class CompositeModel(Model):
    """Sum or product of two models; created with ``model_a + model_b`` or ``model_a * model_b``.

    Parameter names that occur in more than one component get automatic prefixes, for example
    ``gaussian + gaussian`` has parameters ``g1_amp, g1_cen, g1_sigma, g2_amp, ...``. Components
    with an explicit prefix (:meth:`Model.with_prefix`) keep it.
    """

    def __init__(self, left: Model, right: Model, op: str) -> None:
        if op not in _OPS:
            raise ValueError(f"unsupported operator {op!r}")
        leaves = _resolve_prefixes([*left.leaves, *right.leaves])
        it = iter(leaves)
        self._init_parts(left._replace_leaves(it), right._replace_leaves(it), op)

    def _init_parts(self, left: Model, right: Model, op: str) -> None:
        self.left, self.right, self.op = left, right, op
        names = left.param_names + right.param_names
        dupes = sorted(n for n, c in Counter(names).items() if c > 1)
        if dupes:
            raise ValueError(
                f"parameter names {dupes} occur in several components; "
                "give the components distinct prefixes with .with_prefix()"
            )
        self._names = names
        self.name = f"{left.name}{op}{right.name}"
        self.prefix = ""
        self._auto_prefix = False
        self.symmetric = frozenset()
        self.shape = "other"
        self.formula = None
        self.param_docs = {**left.param_docs, **right.param_docs}
        self.guess_label = "composite"
        self.abbrev = "c"
        self.linear = _combined_linearity(left.linear, right.linear, op)

    def _replace_leaves(self, it: Iterator[Model]) -> Model:
        new = CompositeModel.__new__(CompositeModel)
        new._init_parts(self.left._replace_leaves(it), self.right._replace_leaves(it), self.op)
        return new

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names of all components."""
        return self._names

    @property
    def defaults(self) -> dict[str, float]:
        """Starting values from the component signatures."""
        return {**self.left.defaults, **self.right.defaults}

    @property
    def leaves(self) -> tuple[Model, ...]:
        """The elementary models in evaluation order."""
        return self.left.leaves + self.right.leaves

    def with_prefix(self, prefix: str) -> Model:
        """Not supported; prefix the components instead."""
        raise TypeError("prefix the components of a composite model, not the composite")

    def eval(self, x: ArrayLike, params: Mapping[str, float]) -> FloatArray:
        """Evaluate both components and combine them."""
        return _OPS[self.op](self.left.eval(x, params), self.right.eval(x, params))

    def _guess_labeled(self, x: FloatArray, y: FloatArray) -> dict[str, tuple[float, str]]:
        if self._is_pure_sum() and x.ndim == 1:
            return _guess_sum(self.leaves, x, y)
        out: dict[str, tuple[float, str]] = {}
        for leaf in self.leaves:
            out.update(leaf._guess_labeled(x, y))
        return out

    def _is_pure_sum(self) -> bool:
        parts = (self.left, self.right)
        return self.op == "+" and all(
            not isinstance(p, CompositeModel) or p._is_pure_sum() for p in parts
        )

    def signature(self) -> str:
        """Signatures of the components joined by the operator."""

        def part(m: Model) -> str:
            s = m.signature()
            return f"({s})" if isinstance(m, CompositeModel) and m.op != self.op else s

        return f"{part(self.left)} {self.op} {part(self.right)}"

    def __repr__(self) -> str:
        return f"CompositeModel({self.signature()})"


def as_model(obj: Model | Callable[..., ArrayLike] | str) -> Model:
    """Convert a function or expression string to a :class:`Model`; models pass through."""
    if isinstance(obj, Model):
        return obj
    if isinstance(obj, str):
        from fitwise.expression import ExpressionModel

        return ExpressionModel(obj)
    if callable(obj):
        return Model(obj)
    raise TypeError(f"expected a Model or a function, got {type(obj).__name__}")


# --- helpers ------------------------------------------------------------------------------


def _introspect(
    func: Callable[..., Any], param_names: Sequence[str] | None
) -> tuple[tuple[str, ...], dict[str, float]]:
    if param_names is not None:
        names = tuple(param_names)
        defaults: dict[str, float] = {}
    else:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            raise TypeError(
                f"cannot read the signature of {func!r}; pass param_names=[...]"
            ) from None
        params = list(sig.parameters.values())
        if not params:
            raise TypeError("the model function needs the independent variable as first argument")
        positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params:
            if p.kind not in positional:
                raise TypeError(
                    f"cannot infer parameters from {p.kind.description} argument {p.name!r}; "
                    "use plain positional arguments f(x, a, b) or pass param_names=[...]"
                )
        names = tuple(p.name for p in params[1:])
        defaults = {}
        for p in params[1:]:
            if p.default is not inspect.Parameter.empty:
                try:
                    defaults[p.name] = float(p.default)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"default of parameter {p.name!r} must be a number, got {p.default!r}"
                    ) from None
    if not names:
        raise TypeError("the model has no parameters to fit")
    for n in names:
        if not n.isidentifier() or keyword.iskeyword(n):
            raise ValueError(f"parameter name {n!r} is not a valid identifier")
    dupes = sorted(n for n, c in Counter(names).items() if c > 1)
    if dupes:
        raise ValueError(f"duplicate parameter names: {dupes}")
    return names, defaults


def _bind(model: Model, args: Sequence[float], kwargs: Mapping[str, float]) -> dict[str, float]:
    names = model.param_names
    if len(args) > len(names):
        raise TypeError(f"{model.name} takes {len(names)} parameters, got {len(args)}")
    values = {**model.defaults, **dict(zip(names, args, strict=False))}
    for k, v in kwargs.items():
        if k not in names:
            raise TypeError(f"{model.name} has no parameter {k!r}")
        if k in names[: len(args)]:
            raise TypeError(f"parameter {k!r} given twice")
        values[k] = v
    missing = [n for n in names if n not in values]
    if missing:
        raise TypeError(f"missing parameter values: {', '.join(missing)}")
    return values


def _resolve_prefixes(leaves: list[Model]) -> list[Model]:
    leaves = [leaf._without_auto_prefix() for leaf in leaves]
    counts = Counter(n for leaf in leaves for n in leaf.param_names)
    clashing = [
        i
        for i, leaf in enumerate(leaves)
        if not leaf.prefix and any(counts[n] > 1 for n in leaf.param_names)
    ]
    per_abbrev = Counter(leaves[i].abbrev for i in clashing)
    seen: Counter[str] = Counter()
    result = list(leaves)
    for i in clashing:
        abbrev = leaves[i].abbrev
        seen[abbrev] += 1
        tag = f"{abbrev}{seen[abbrev]}" if per_abbrev[abbrev] > 1 else abbrev
        result[i] = leaves[i]._prefixed(f"{tag}_", auto=True)
    return result


def _combined_linearity(a: bool | None, b: bool | None, op: str) -> bool | None:
    if op == "*":
        return False
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _guess_sum(
    leaves: Sequence[Model], x: FloatArray, y: FloatArray
) -> dict[str, tuple[float, str]]:
    """Guess a sum of models sequentially: baselines first, then signals on the residual."""
    order = np.argsort(x, kind="stable")
    xs, resid = x[order], y[order].copy()
    baselines = [m for m in leaves if m.shape == "baseline"]
    signals = [m for m in leaves if m.shape != "baseline"]
    region = _baseline_region(signals[0].shape, xs, resid) if baselines and signals else slice(None)

    out: dict[str, tuple[float, str]] = {}
    for leaf, sub in [*((b, region) for b in baselines), *((s, slice(None)) for s in signals)]:
        guess = leaf._guess_labeled(xs[sub], resid[sub])
        if baselines and signals and leaf.shape == "baseline":
            guess = {n: (v, "edge baseline") for n, (v, _) in guess.items()}
        out.update(guess)
        values = {n: v for n, (v, _) in guess.items()}
        if all(n in values for n in leaf.param_names):
            with np.errstate(all="ignore"):
                contribution = leaf.eval(xs, values)
            if np.all(np.isfinite(contribution)):
                resid -= contribution
    return out


def _baseline_region(shape: Shape, xs: FloatArray, ys: FloatArray) -> slice | NDArray[np.intp]:
    n = len(xs)
    m = max(3, n // 6)
    if shape == "oscillation" or 2 * m >= n:
        return slice(None)
    left, right = np.arange(m), np.arange(n - m, n)
    if shape == "monotone":
        return left if np.ptp(ys[left]) < np.ptp(ys[right]) else right
    return np.concatenate([left, right])
