"""Shared fixtures for the Python benchmarks."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so benchmark inputs are identical across runs."""
    return np.random.default_rng(2026)
