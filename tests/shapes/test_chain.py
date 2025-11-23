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


def test_chain_preserves_neighbor_consistency_in_2d(
    structured_chain_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Sanity Check (2D -> 2D): Tests that neighbor distances in the projection are consistent.
    """
    X, y = structured_chain_data_2d
    X_proj = smds_engine.fit_transform(X, y)

    # Check the first few links in the chain. They should all be roughly the same length (1.0).
    dist_0_1 = np.linalg.norm(X_proj[0] - X_proj[1])
    dist_1_2 = np.linalg.norm(X_proj[1] - X_proj[2])
    dist_2_3 = np.linalg.norm(X_proj[2] - X_proj[3])

    # Allow for 20% relative variation due to optimization noise
    assert np.isclose(dist_0_1, dist_1_2, rtol=0.2), (
        f"Inconsistent chain links: Link(0-1)={dist_0_1:.4f} vs Link(1-2)={dist_1_2:.4f}"
    )
    assert np.isclose(dist_1_2, dist_2_3, rtol=0.2), (
        f"Inconsistent chain links: Link(1-2)={dist_1_2:.4f} vs Link(2-3)={dist_2_3:.4f}"
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

    # Compares the recovered shape to the original latent shape, ignoring rotation/scale.
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)
    procrustes_threshold = 0.1
    assert disparity < procrustes_threshold, (
        f"Shape recovery failed Procrustes analysis. "
        f"The disparity ({disparity:.4f}) exceeds the threshold of {procrustes_threshold}."
    )

    # The score measures how well the projection satisfies the ChainShape distance rules.
    score = smds_engine.score(X, y)
    score_threshold = 0.95
    assert score > score_threshold, (
        f"The SMDS score is too low. Expected a score greater than {score_threshold}, but got {score:.4f}."
    )