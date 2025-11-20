import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray
from scipy.spatial import procrustes

from smds import SupervisedMDS
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape


def test_discrete_circular_init_validation() -> None:
    """Tests that the __init__ method validates its parameters correctly."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        DiscreteCircularShape(num_points=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        DiscreteCircularShape(num_points=-5)

    try:
        DiscreteCircularShape(num_points=12)
    except ValueError as e:
        pytest.fail(f"Initialization with a valid positive integer failed: {e}")


def test_discrete_circular_input_validation() -> None:
    """Tests that the _validate_input method raises correct errors."""
    shape = DiscreteCircularShape(num_points=12)

    with pytest.raises(ValueError, match="must be 1-dimensional"):
        invalid_2d_input: NDArray[np.float64] = np.array([[1.0, 2.0], [3.0, 4.0]])
        shape(invalid_2d_input)

    with pytest.raises(ValueError, match="cannot be empty"):
        empty_input: NDArray[np.float64] = np.array([])
        shape(empty_input)


def test_discrete_circular_distance_computation() -> None:
    """Tests the core wrap-around distance logic with a hand-calculated example."""
    shape = DiscreteCircularShape(num_points=12)

    y: NDArray[np.float64] = np.array([0, 1, 6, 11]).astype(float)

    dists: NDArray[np.float64] = shape(y)

    expected: NDArray[np.float64] = np.array(
        [
            [0.0, 1.0, 6.0, 1.0],
            [1.0, 0.0, 5.0, 2.0],
            [6.0, 5.0, 0.0, 5.0],
            [1.0, 2.0, 5.0, 0.0],
        ]
    )

    assert_array_almost_equal(
        dists, expected, err_msg="The computed distance matrix does not match the expected wrap-around distances."
    )


@pytest.fixture
def random_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Provides random X and y data for basic testing."""
    X: NDArray[np.float64] = np.random.randn(50, 10)
    y: NDArray[np.float64] = np.random.randint(0, 12, size=50).astype(float)
    return X, y


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    """Provides a default SMDS engine configured for a 12-point circular shape."""
    return SupervisedMDS(n_components=2, manifold=DiscreteCircularShape(num_points=12))


@pytest.mark.smoke
def test_discrete_circular_smoke_test(
    random_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """Smoke Test: Ensures the class can be fit/transformed without errors."""
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = random_data

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    n_samples = X.shape[0]
    n_components = smds_engine.n_components
    assert X_proj.shape == (n_samples, n_components), (
        f"Output shape is incorrect. Expected {(n_samples, n_components)}, but got {X_proj.shape}."
    )


@pytest.fixture
def structured_circular_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Provides a simple 2D dataset that is already structured as a circle."""
    n_labels: int = 12
    points_per_label: int = 4
    n_samples: int = n_labels * points_per_label

    y: NDArray[np.float64] = np.repeat(np.arange(n_labels), points_per_label).astype(float)

    angles: NDArray[np.float64] = 2 * np.pi * y / n_labels
    X: NDArray[np.float64] = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X += np.random.randn(n_samples, 2) * 0.1

    indices: NDArray[np.int_] = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]


def test_discrete_circular_preserves_structure_in_2d(
    structured_circular_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """Sanity Check (2D -> 2D): Tests if adjacent points are closer than opposite points."""
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = structured_circular_data_2d

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    n_labels = smds_engine.manifold.num_points
    indices_0 = np.where(y == 0.0)[0]
    indices_1 = np.where(y == 1.0)[0]
    indices_opposite = np.where(y == n_labels // 2)[0]

    centroid_0 = X_proj[indices_0].mean(axis=0)
    centroid_1 = X_proj[indices_1].mean(axis=0)
    centroid_opposite = X_proj[indices_opposite].mean(axis=0)

    dist_adjacent = np.linalg.norm(centroid_0 - centroid_1)
    dist_opposite = np.linalg.norm(centroid_0 - centroid_opposite)

    assert dist_adjacent < dist_opposite, (
        f"Adjacent points are not closer than opposite points. "
        f"Adjacent dist: {dist_adjacent:.4f}, Opposite dist: {dist_opposite:.4f}."
    )


@pytest.fixture
def structured_circular_data_high_dim(
    structured_circular_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Provides a high-dimensional dataset containing a hidden 2D circular structure."""
    X_latent, y = structured_circular_data_2d
    latent_dim: int = 2
    high_dim: int = 10

    projection_matrix: NDArray[np.float64] = np.random.randn(latent_dim, high_dim)
    X_high_dim: NDArray[np.float64] = X_latent @ projection_matrix

    return X_high_dim, y, X_latent


def test_discrete_circular_recovers_structure_from_high_dim(
    structured_circular_data_high_dim: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """Advanced Test (10D -> 2D): Tests if SMDS can recover a circular structure."""
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X_original: NDArray[np.float64]
    X, y, X_original = structured_circular_data_high_dim

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    # Compares the recovered shape to the original latent circle.
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)
    procrustes_threshold = 0.2
    assert disparity < procrustes_threshold, (
        f"Shape recovery failed Procrustes analysis. "
        f"The disparity ({disparity:.4f}) exceeds the threshold of {procrustes_threshold}."
    )

    # The score measures how well the projection satisfies the circular distance rules.
    score = smds_engine.score(X, y)
    score_threshold = 0.9
    assert score > score_threshold, (
        f"The SMDS score is too low. Expected a score greater than {score_threshold}, but got {score:.4f}."
    )
