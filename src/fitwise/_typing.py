"""Shared type aliases."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Bounds: TypeAlias = tuple[float, float]

__all__ = ["ArrayLike", "Bounds", "FloatArray"]
