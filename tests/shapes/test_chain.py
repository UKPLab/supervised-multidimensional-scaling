import numpy as np
import pytest
from numpy.typing import NDArray
from numpy.testing import assert_array_almost_equal
from scipy.spatial import procrustes

from smds import SupervisedMDS
from smds.shapes.discrete_shapes.chain import ChainShape


def test_chain_init_validation() -> None:
    """Tests that the __init__ method validates its parameters correctly."""
    with pytest.raises(ValueError, match="threshold must be positive"):
        ChainShape(threshold=0)


def test_chain_distance_computation() -> None:
    """Tests the core neighbor-only wrap-around distance logic."""
    shape = ChainShape(threshold=2.0)
    y: NDArray[np.float64] = np.array([0, 1, 2, 3]).astype(float)
    dists: NDArray[np.float64] = shape(y)
    expected: NDArray[np.float64] = np.array([
        [0.0, 1.0, -1.0, 1.0], [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0],
    ])
    assert_array_almost_equal(dists, expected)


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    """Provides a default SMDS engine configured with ChainShape."""
    return SupervisedMDS(n_components=2, manifold=ChainShape(threshold=1.1))


@pytest.fixture
def structured_chain_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a 2D dataset arranged in a circular pattern to reflect the cyclical structure of ChainShape.
    """
    n_points = 20
    y: NDArray[np.float64] = np.arange(n_points).astype(float)

    # Create points on a unit circle
    angles = 2 * np.pi * y / n_points
    X: NDArray[np.float64] = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Add noise
    X += np.random.randn(n_points, 2) * 0.01

    return X, y



@pytest.mark.smoke
def test_chain_smoke_test(
        structured_chain_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
        smds_engine: SupervisedMDS,
) -> None:
    """Smoke Test: Ensures the class can be fit/transformed without errors."""
    X, y = structured_chain_data_2d
    X_proj = smds_engine.fit_transform(X, y)
    assert X_proj.shape == (X.shape[0], smds_engine.n_components), "Output shape is incorrect."

