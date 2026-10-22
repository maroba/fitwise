import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def peak_data(rng):
    """Gaussian peak on a sloped background with known noise."""
    x = np.linspace(-5, 5, 128)
    y = 2.33 * np.exp(-((x - 0.21) ** 2) / (2 * 1.51**2)) - 0.003 * x + 0.1
    dy = np.full_like(x, 0.02)
    return x, y + rng.normal(0, 0.02, x.size), dy
