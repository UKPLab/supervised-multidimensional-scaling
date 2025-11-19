import numpy as np
import pytest
from numpy.typing import NDArray
from smds import SupervisedMDS

from smds.shapes.discrete_shapes.hierarchical import HierarchicalShape
from smds.shapes.discrete_shapes.cluster import ClusterShape


# =============================================================================
# Helper
# =============================================================================

def _compare_shape_scores(
        X: NDArray[np.float64],
        y_hierarchical: NDArray[np.float64],
        engine_hierarchical: SupervisedMDS,
        engine_cluster: SupervisedMDS,
) -> tuple[float, float]:
    """
    Fits and scores both hierarchical and cluster engines on the same data.

    Returns:
        A tuple of (hierarchical_score, cluster_score).
    """
    # Score the HierarchicalShape
    engine_hierarchical.fit(X, y_hierarchical)
    hierarchical_score = engine_hierarchical.score(X, y_hierarchical)

    # Score the ClusterShape after flattening the labels
    _, y_flat = np.unique(y_hierarchical, axis=0, return_inverse=True)
    engine_cluster.fit(X, y_flat)
    cluster_score = engine_cluster.score(X, y_flat)

    return hierarchical_score, cluster_score


@pytest.fixture
def smds_engine_hierarchical_new() -> SupervisedMDS:
    """Provides an SMDS engine configured for a 3-level hierarchy."""
    return SupervisedMDS(n_components=2, manifold=HierarchicalShape([100.0, 10.0, 1.0]))


@pytest.fixture
def smds_engine_hierachical_old() -> SupervisedMDS:
    """Provides an SMDS engine configured for a 3-level hierarchy."""
    return SupervisedMDS(n_components=2, manifold=HierarchicalShape(level_distances=[1.0, 2.0, 3.0]))


@pytest.fixture
def smds_engine_cluster() -> SupervisedMDS:
    """Provides a generic SMDS engine configured with ClusterShape."""
    return SupervisedMDS(n_components=2, manifold=ClusterShape())


# =============================================================================
# Original Test
# =============================================================================

@pytest.fixture
def structured_hierarchical_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a dataset with clear hierarchical structure:
    - Level 0: Two main groups (0 and 1)
    - Level 1: Subgroups within each main group
    - Level 2: Fine-grained distinctions
    """
    n_per_group = 25

    group0_sub0 = np.random.randn(n_per_group, 2) + np.array([-10, -10])
    group0_sub1 = np.random.randn(n_per_group, 2) + np.array([-10, 10])
    group1_sub0 = np.random.randn(n_per_group, 2) + np.array([10, -10])
    group1_sub1 = np.random.randn(n_per_group, 2) + np.array([10, 10])

    X: NDArray[np.float64] = np.vstack([group0_sub0, group0_sub1, group1_sub0, group1_sub1])

    y: NDArray[np.float64] = np.array(
        [[0, 0, 0]] * n_per_group +
        [[0, 0, 1]] * n_per_group +
        [[1, 0, 0]] * n_per_group +
        [[1, 0, 1]] * n_per_group
    ).astype(float)

    indices: NDArray[np.int_] = np.arange(X.shape[0])
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_cluster_outperforms_hierarchical_on_old_data(
        structured_hierarchical_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
        smds_engine_hierachical_old: SupervisedMDS,
        smds_engine_cluster: SupervisedMDS,
) -> None:
    """
    On data with 4 simple blobs (where geometry ignores hierarchy),
    ClusterShape must achieve a higher score than HierarchicalShape.
    """
    X, y_hierarchical = structured_hierarchical_data_2d

    hierarchical_score, cluster_score = _compare_shape_scores(
        X, y_hierarchical, smds_engine_hierachical_old, smds_engine_cluster
    )

    print(f"On old data -> Hierarchical Score: {hierarchical_score:.4f}, Cluster Score: {cluster_score:.4f}")

    assert cluster_score > hierarchical_score, (
        f"ClusterShape score ({cluster_score:.4f}) was not higher than "
        f"HierarchicalShape score ({hierarchical_score:.4f}) on a simple old dataset."
    )


# =============================================================================
# New Test
# =============================================================================

@pytest.fixture
def new_hierarchical_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides data where the X geometry newLY matches the Y hierarchy.
    HierarchicalShape should excel here.
    """
    n_examples = 10
    y_list = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 7]]
    y = np.repeat(y_list, n_examples, axis=0).astype(float)

    level_1_offset = np.array([-50, 0]) * (1 - y[:, 0:1]) + np.array([50, 0]) * y[:, 0:1]
    level_2_offset = np.array([0, -10]) * (1 - y[:, 1:2]) + np.array([0, 10]) * y[:, 1:2]
    level_3_offset = np.random.randn(y.shape[0], 2) * 1.0
    X = level_1_offset + level_2_offset + level_3_offset

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def test_hierarchical_outperforms_cluster_on_new_data(
        new_hierarchical_data,
        smds_engine_hierarchical_new: SupervisedMDS,
        smds_engine_cluster: SupervisedMDS,
) -> None:
    """
    On data where the geometry newly matches the hierarchy,
    HierarchicalShape must achieve a higher score than ClusterShape.
    """
    X, y_hierarchical = new_hierarchical_data

    hierarchical_score, cluster_score = _compare_shape_scores(
        X, y_hierarchical, smds_engine_hierarchical_new, smds_engine_cluster
    )

    print(f"On new data -> Hierarchical Score: {hierarchical_score:.4f}, Cluster Score: {cluster_score:.4f}")

    assert hierarchical_score > cluster_score, (
        f"HierarchicalShape score ({hierarchical_score:.4f}) was not higher than "
        f"ClusterShape score ({cluster_score:.4f}) on a newly hierarchical dataset."
    )
    assert hierarchical_score > 0.9, f"Hierarchical score is unexpectedly low ({hierarchical_score:.4f})"

