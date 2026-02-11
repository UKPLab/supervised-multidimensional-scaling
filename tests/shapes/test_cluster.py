import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from smds import SupervisedMDS
from smds.shapes.discrete_shapes import ClusterShape


def test_cluster_input_validation() -> None:
    """Tests that the _validate_input method raises correct errors."""
    shape = ClusterShape()

    # 2D Input
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        invalid_2d_input: NDArray[np.float64] = np.array([[1.0, 2.0], [3.0, 4.0]])
        shape(invalid_2d_input)

    # Empty Input
    with pytest.raises(ValueError, match="cannot be empty"):
        empty_input: NDArray[np.float64] = np.array([])
        shape(empty_input)


def test_cluster_distance_computation() -> None:
    """
    Tests the core logic: Distances should be 0 for same label, 1 for different.
    """
    shape = ClusterShape()
    y: NDArray[np.float64] = np.array([0, 0, 1, 2]).astype(float)

    dists: NDArray[np.float64] = shape(y)

    expected: NDArray[np.float64] = np.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    )

    assert_array_equal(
        dists, expected, err_msg="Cluster distance logic failed. Expected 0 for same class, 1 for different."
    )


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    """Provides a default SMDS engine configured with ClusterShape."""
    return SupervisedMDS(stage_1="computed", manifold="cluster")


@pytest.fixture
def structured_cluster_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a dataset where points are already clearly separated into two groups.
    """
    n_per_cluster = 25

    # Create 25 points centered around [-10, -10]
    cluster0: NDArray[np.float64] = np.random.randn(n_per_cluster, 2) - 10

    # Create 25 points centered around [10, 10]
    cluster1: NDArray[np.float64] = np.random.randn(n_per_cluster, 2) + 10

    X: NDArray[np.float64] = np.vstack([cluster0, cluster1])
    y: NDArray[np.float64] = np.array([0.0] * n_per_cluster + [1.0] * n_per_cluster)

    # Shuffle the data
    indices: NDArray[np.int_] = np.arange(X.shape[0])
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_cluster_preserves_separation_in_2d(
    structured_cluster_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Sanity Check (2D -> 2D): Tests that clusters remain separated.
    Logic: Average intra-cluster distance < Distance between centroids.
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

    # Calculate the distance between the centroids
    centroid_c0 = X_proj[indices_c0].mean(axis=0)
    centroid_c1 = X_proj[indices_c1].mean(axis=0)
    dist_between_clusters = np.linalg.norm(centroid_c0 - centroid_c1)

    assert avg_within_cluster_dist < dist_between_clusters, (
        f"Clusters are not well-separated. "
        f"Average within-cluster distance ({avg_within_cluster_dist:.4f}) "
        f"should be less than the distance between centroids ({dist_between_clusters:.4f})."
    )
