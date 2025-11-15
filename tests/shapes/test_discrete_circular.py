import numpy as np
import pytest
from scipy.spatial import procrustes

from smds import SupervisedMDS
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape  # Adjusted import path


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    """Provides random X and y data for basic testing."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 12, size=50).astype(float)
    return X, y


def test_discrete_circular_smoke_test(random_data: tuple[np.ndarray, np.ndarray]) -> None:
    """
    A simple "smoke test" to ensure DiscreteCircularShape can be fit and
    transformed without errors and produces an output of the correct shape.
    """
    X, y = random_data
    n_samples = X.shape[0]
    n_components = 2

    smds_engine = SupervisedMDS(n_components=n_components, manifold=DiscreteCircularShape())
    X_proj = smds_engine.fit_transform(X, y)

    assert X_proj.shape == (n_samples, n_components)


@pytest.fixture
def structured_circular_data_2d() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides a simple 2D dataset that is already perfectly structured as a circle.
    """
    n_labels = 12
    points_per_label = 4
    n_samples = n_labels * points_per_label

    y = np.repeat(np.arange(n_labels), points_per_label).astype(float)

    angles = 2 * np.pi * y / n_labels
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X += np.random.randn(n_samples, 2) * 0.1  # Add a little noise

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_discrete_circular_preserves_structure_in_2d(
        structured_circular_data_2d: tuple[np.ndarray, np.ndarray]) -> None:
    """
    Sanity Check (2D -> 2D): Tests the core behavior of DiscreteCircularShape: ensuring that
    points with adjacent labels are closer than points with opposite labels.
    """
    X, y = structured_circular_data_2d
    smds_engine = SupervisedMDS(n_components=2, manifold=DiscreteCircularShape())
    X_proj = smds_engine.fit_transform(X, y)

    # Assert that adjacent points are closer than opposite points
    n_labels = int(np.max(y) + 1)
    indices_0 = np.where(y == 0.0)[0]
    indices_1 = np.where(y == 1.0)[0]
    indices_opposite = np.where(y == n_labels // 2)[0]

    centroid_0 = X_proj[indices_0].mean(axis=0)
    centroid_1 = X_proj[indices_1].mean(axis=0)
    centroid_opposite = X_proj[indices_opposite].mean(axis=0)

    dist_adjacent = np.linalg.norm(centroid_0 - centroid_1)
    dist_opposite = np.linalg.norm(centroid_0 - centroid_opposite)

    assert dist_adjacent < dist_opposite, "Adjacent points are not closer than opposite points."


# ToDo: test for int labels IF team decides to allow non-float types.

@pytest.fixture
def structured_circular_data_high_dim() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Provides a high-dimensional dataset containing a hidden 2D circular structure.
    This tests the ability of SMDS to recover the latent circular manifold.
    """
    n_labels = 12
    points_per_label = 4
    n_samples = n_labels * points_per_label
    latent_dim = 2
    high_dim = 10

    # Create the labels
    y = np.repeat(np.arange(n_labels), points_per_label).astype(float)

    # Create the simple, 2D ground-truth data (points on a circle)
    angles = 2 * np.pi * y / n_labels
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Add a small amount of noise to make them a "cloudy" circle
    X_latent += np.random.randn(n_samples, latent_dim) * 0.1

    # Create a random projection matrix to map into the high-dimensional space
    projection_matrix = np.random.randn(latent_dim, high_dim)
    X_high_dim = X_latent @ projection_matrix

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Return the high-dim data, the labels, and the original 2D structure
    return X_high_dim[indices], y[indices], X_latent[indices]


def test_discrete_circular_recovers_structure_from_high_dim(
        structured_circular_data_high_dim: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """
    Tests if SMDS can recover a 2D discrete circular structure from a high-dimensional space.
    """
    X_high_dim, y, X_original = structured_circular_data_high_dim

    smds_engine = SupervisedMDS(n_components=2, manifold=DiscreteCircularShape())
    X_proj = smds_engine.fit_transform(X_high_dim, y)

    # --- Sanity check: Test order preservation ---
    # Points with adjacent labels (e.g., 0 and 1) should be close.
    # Points with opposite labels (e.g., 0 and 6) should be far.
    indices_0 = np.where(y == 0.0)[0]
    indices_1 = np.where(y == 1.0)[0]
    indices_6 = np.where(y == 6.0)[0]  # Opposite point on a 12-point circle

    centroid_0 = X_proj[indices_0].mean(axis=0)
    centroid_1 = X_proj[indices_1].mean(axis=0)
    centroid_6 = X_proj[indices_6].mean(axis=0)

    dist_adjacent = np.linalg.norm(centroid_0 - centroid_1)
    dist_opposite = np.linalg.norm(centroid_0 - centroid_6)

    #print(f"Avg adjacent label distance: {dist_adjacent:.4f}")
    #print(f"Avg opposite label distance: {dist_opposite:.4f}")
    assert dist_adjacent < dist_opposite, "Sanity check failed: Adjacent points are not closer than opposite points."

    # --- Procrustes Analysis ---
    # Compares the recovered shape to the original latent circle, ignoring rotation/scale.
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)
    #print(f"Procrustes disparity between original and recovered shape: {disparity:.4f}")
    assert disparity < 0.2, "Procrustes analysis shows the recovered shape is too different from the original circle."

    # --- Built-in Score ---
    # The score measures how well the projection satisfies the circular distance rules.
    score = smds_engine.score(X_high_dim, y)
    print(f"SMDS score for DiscreteCircular: {score:.4f}")
    assert score > 0.9, "SMDS score is too low, indicating a poor circular recovery."
