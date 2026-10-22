"""Shared pytest setup for tests and doctests."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    doctest_namespace["np"] = np
