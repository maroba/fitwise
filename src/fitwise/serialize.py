"""JSON-serializable records of fits, for reproducibility and pipelines."""

from __future__ import annotations

import hashlib
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fitwise.result import FitResult

__all__ = ["SCHEMA_VERSION", "data_hash", "to_dict"]

#: Version of the dictionary layout produced by :func:`to_dict`.
SCHEMA_VERSION = 1


def _float(v: float | None) -> float | str | None:
    """JSON has no inf/nan: map them to strings, keep None."""
    if v is None:
        return None
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(v)


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def data_hash(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray | None, xerr: np.ndarray | None = None
) -> str:
    """SHA-256 over the data arrays (shape, dtype and bytes), to recognize identical data.

    ``xerr`` only enters if given, so hashes of fits without it are the same as in 0.1.
    """
    h = hashlib.sha256()
    for arr in (x, y, yerr, *(() if xerr is None else (xerr,))):
        if arr is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def to_dict(result: FitResult, *, include_data: bool = False) -> dict[str, Any]:
    """Describe a fit completely in JSON-compatible types.

    Parameters
    ----------
    result : FitResult
        The fit to serialize.
    include_data : bool
        Also store ``x``, ``y`` and ``yerr``. Otherwise only their hash is stored.

    Returns
    -------
    dict
        With keys ``schema_version``, ``versions``, ``model``, ``data``, ``options``,
        ``method``, ``converged``, ``parameters``, ``covariance``, ``statistics`` and
        ``warnings``.
    """
    data: dict[str, Any] = {
        "n": result.data.n,
        "n_omitted": result.data.n_omitted,
        "has_yerr": result.yerr is not None,
        "has_xerr": result.data.xerr is not None,
        "sha256": data_hash(result.x, result.y, result.yerr, result.data.xerr),
    }
    if include_data:
        data["x"] = result.x.tolist()
        data["y"] = result.y.tolist()
        data["yerr"] = None if result.yerr is None else result.yerr.tolist()
        data["xerr"] = None if result.data.xerr is None else result.data.xerr.tolist()

    return {
        "schema_version": SCHEMA_VERSION,
        "versions": {
            "fitwise": _version("fitwise"),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "python": platform.python_version(),
        },
        "model": {
            "signature": result.model.signature(),
            "param_names": list(result.model.param_names),
            "components": [leaf.name for leaf in result.model.leaves],
        },
        "data": data,
        "options": {
            k: ({n: [_float(b) for b in v] for n, v in val.items()} if k == "bounds" else val)
            for k, val in result.options.items()
        },
        "method": {
            "solver": result.method,
            "reason": result.method_reason,
            "nfev": result.nfev,
            "message": result.message,
            "log": list(result.solver_log),
        },
        "converged": result.converged,
        "ok": result.ok,
        "parameters": {
            name: {
                "value": _float(p.value),
                "stderr": _float(p.stderr),
                "init_value": _float(p.init_value),
                "guess_strategy": p.guess_strategy,
                "bounds": [_float(p.bounds[0]), _float(p.bounds[1])],
                "fixed": p.fixed,
            }
            for name, p in result.params.items()
        },
        "covariance": {
            "names": list(result.free_names),
            "scaled": result.scale_covar,
            "matrix": None
            if result.covariance is None
            else [[_float(v) for v in row] for row in result.covariance.tolist()],
        },
        "statistics": {
            "chi2": _float(result.chi2),
            "ndf": result.ndf,
            "redchi": _float(result.redchi),
            "pvalue": _float(result.pvalue),
            "rss": _float(result.rss),
            "residual_sd": _float(result.residual_sd),
            "r2": _float(result.r2),
            "aic": _float(result.aic),
            "bic": _float(result.bic),
        },
        "warnings": [{"type": type(w).__name__, "message": str(w)} for w in result.warnings],
        "robust": None
        if result.robust_weights is None
        else {
            "loss": result.loss,
            "n_downweighted": int(result.downweighted.sum()),
            **({"weights": result.robust_weights.tolist()} if include_data else {}),
        },
    }
