import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist

from smds import SupervisedMDS
from smds.shapes.discrete_shapes.cluster import ClusterShape


@pytest.fixture
def random_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Provides random X and y data for basic testing."""
    np.random.seed(2137)
    X: NDArray[np.float64] = np.random.randn(50, 10)
    y: NDArray[np.float64] = np.random.randint(0, 3, size=50).astype(float)
    return X, y


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    """Provides a default SupervisedMDS engine for testing."""
    return SupervisedMDS(n_components=2, manifold=ClusterShape())


@pytest.mark.smoke
def test_cluster_smoke_test(
    random_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    A simple "smoke test" to ensure ClusterShape can be fit and transformed
    without errors and produces an output of the correct shape.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = random_data

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    n_samples = X.shape[0]
    n_components = smds_engine.n_components

    assert X_proj.shape == (n_samples, n_components), (
        f"Output shape is incorrect. "
        f"Expected (n_samples, n_components): {(n_samples, n_components)}, "
        f"but got: {X_proj.shape}."
    )


@pytest.fixture
def structured_cluster_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a dataset where points are already clearly separated into two groups.
    """
    # Create 25 points centered around [-10, -10]
    cluster0: NDArray[np.float64] = np.random.randn(25, 2) - 10

    # Create 25 points centered around [10, 10]
    cluster1: NDArray[np.float64] = np.random.randn(25, 2) + 10

    X: NDArray[np.float64] = np.vstack([cluster0, cluster1])
    y: NDArray[np.float64] = np.array([0.0] * 25 + [1.0] * 25)

    # Shuffle the data
    indices: NDArray[np.int_] = np.arange(50)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_cluster_preserves_structure_in_2d(
    structured_cluster_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Sanity Check (2D -> 2D): Tests the core behavior of ClusterShape: ensuring that points with the same
    label are closer to each other than to points with different labels.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = structured_cluster_data_2d

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    # Identify the indices for each cluster
    indices_c0 = np.where(y == 0.0)[0]
    indices_c1 = np.where(y == 1.0)[0]

    # Calculate the average within-cluster distance
    dist_within_c0 = pdist(X_proj[indices_c0]).mean()
    dist_within_c1 = pdist(X_proj[indices_c1]).mean()
    avg_within_cluster_dist = (dist_within_c0 + dist_within_c1) / 2

    # Calculate the distance between the centroids of the two clusters
    centroid_c0 = X_proj[indices_c0].mean(axis=0)
    centroid_c1 = X_proj[indices_c1].mean(axis=0)
    dist_between_clusters = np.linalg.norm(centroid_c0 - centroid_c1)

    assert avg_within_cluster_dist < dist_between_clusters, (
        f"Clusters are not well-separated. "
        f"Average within-cluster distance ({avg_within_cluster_dist:.4f}) "
        f"should be less than the distance between centroids ({dist_between_clusters:.4f})."
    )
