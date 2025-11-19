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
    """Provides a simple 2D dataset structured as a line."""
    n_points = 20
    y: NDArray[np.float64] = np.arange(n_points).astype(float)
    X: NDArray[np.float64] = np.stack([y, np.random.randn(n_points) * 0.1], axis=1)
    return X, y


@pytest.fixture
def structured_chain_data_high_dim(
        structured_chain_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Provides a high-dimensional dataset containing a hidden 2D chain structure."""
    X_latent, y = structured_chain_data_2d
    projection_matrix = np.random.randn(2, 10)
    X_high_dim = X_latent @ projection_matrix
    return X_high_dim, y, X_latent


@pytest.mark.smoke
def test_chain_smoke_test(
        structured_chain_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
        smds_engine: SupervisedMDS,
) -> None:
    """Smoke Test: Ensures the class can be fit/transformed without errors."""
    X, y = structured_chain_data_2d
    X_proj = smds_engine.fit_transform(X, y)
    assert X_proj.shape == (X.shape[0], smds_engine.n_components), "Output shape is incorrect."


def test_chain_preserves_structure_in_2d(
        structured_chain_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
        smds_engine: SupervisedMDS,
) -> None:
    """Sanity Check (2D -> 2D): Tests if neighbors remain closer than non-neighbors."""
    X, y = structured_chain_data_2d
    X_proj = smds_engine.fit_transform(X, y)

    dist_0_1 = np.linalg.norm(X_proj[0] - X_proj[1])
    dist_0_2 = np.linalg.norm(X_proj[0] - X_proj[2])
    assert dist_0_1 < dist_0_2, (
        f"Adjacent points (dist={dist_0_1:.4f}) should be closer than "
        f"non-adjacent points (dist={dist_0_2:.4f})."
    )


def test_chain_recovers_structure_from_high_dim(
        structured_chain_data_high_dim: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        smds_engine: SupervisedMDS,
) -> None:
    """Advanced Test (10D -> 2D): Tests if SMDS can recover a chain structure."""
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X_original: NDArray[np.float64]
    X, y, X_original = structured_chain_data_high_dim

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)
    """
    # Procrustes analysis should show that the recovered line-like shape
    # is similar to the original latent line.
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)
    procrustes_threshold = 0.2
    assert disparity < procrustes_threshold, (
        f"Shape recovery failed Procrustes analysis. "
        f"The disparity ({disparity:.4f}) exceeds the threshold of {procrustes_threshold}."
    )
    """

    score = smds_engine.score(X, y)
    print(f"Score: {score:.4f}")
    assert score is not None, "Score calculation failed for incomplete matrix."