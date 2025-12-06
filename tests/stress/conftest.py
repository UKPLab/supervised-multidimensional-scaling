import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]


@pytest.fixture
def original_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generates synthetic high-dimensional and low-dimensional distance vectors."""
    rng = np.random.default_rng(42)
    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")
    return D_high, D_low


@pytest.fixture
def scaled_data(
    original_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns the original data, but with the low-dim distances scaled up significantly."""
    D_high, D_low = original_data
    scale_factor = 1234.5
    D_low_scaled = D_low * scale_factor
    return D_high, D_low_scaled


@pytest.fixture
def perfect_preservation_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generates a case where D_high is identical to D_low (perfect embedding)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    D = pdist(X, metric="euclidean")
    return D, D.copy()
