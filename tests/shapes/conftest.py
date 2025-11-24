import pytest
import numpy as np
from numpy.typing import NDArray


from smds import SupervisedMDS
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape
from smds.shapes.discrete_shapes.hierarchical import HierarchicalShape
from smds.shapes.spiral_shape import SpiralShape


def _project_and_shuffle(
        X_latent: NDArray[np.float64],
        y: NDArray[np.float64],
        high_dim: int = 10,
        noise_level: float = 0.5,
        seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Helper to project latent 2D data into high-dimensional space, add noise,
    and shuffle.
    """
    rng = np.random.default_rng(seed)
    n_samples, latent_dim = X_latent.shape

    # 1. Create random projection matrix
    projection_matrix = rng.standard_normal((latent_dim, high_dim))

    # 2. Project to High Dim
    X_high = X_latent @ projection_matrix

    # 3. Add Noise
    X_high += rng.normal(scale=noise_level, size=X_high.shape)

    # 4. Shuffle indices to ensure model doesn't rely on order
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return X_high[indices], y[indices], X_latent[indices]



# =============================================================================
# CHAIN SETUP (Cyclical topology)
# =============================================================================
@pytest.fixture(scope="module")
def chain_engine():
    return SupervisedMDS(n_components=2, manifold=ChainShape(threshold=1.1))


@pytest.fixture(scope="module")
def chain_data_10d():
    n_points = 20
    # Latent 2D Circle (Matches Chain topology)
    y = np.arange(n_points).astype(float)
    angles = 2 * np.pi * y / n_points
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _project_and_shuffle(X_latent, y)


# =============================================================================
# CLUSTER SETUP
# =============================================================================
@pytest.fixture(scope="module")
def cluster_engine():
    return SupervisedMDS(n_components=2, manifold=ClusterShape())


@pytest.fixture(scope="module")
def cluster_data_10d():
    n_per_cluster = 25
    # Latent 2D Clusters
    c1 = np.random.randn(n_per_cluster, 2) - 10
    c2 = np.random.randn(n_per_cluster, 2) + 10
    X_latent = np.vstack([c1, c2])
    y = np.array([0.0] * n_per_cluster + [1.0] * n_per_cluster)
    return _project_and_shuffle(X_latent, y)


# =============================================================================
# DISCRETE CIRCULAR SETUP
# =============================================================================
@pytest.fixture(scope="module")
def disc_circular_engine():
    return SupervisedMDS(n_components=2, manifold=DiscreteCircularShape(num_points=12))


@pytest.fixture(scope="module")
def disc_circular_data_10d():
    n_labels = 12
    n_per_label = 10
    y = np.repeat(np.arange(n_labels), n_per_label).astype(float)
    # Latent 2D Circle
    angles = 2 * np.pi * y / n_labels
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return _project_and_shuffle(X_latent, y)


# =============================================================================
# HIERARCHICAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def hierarchical_engine():
    return SupervisedMDS(n_components=2, manifold=HierarchicalShape(level_distances=[100.0, 10.0, 1.0]))


@pytest.fixture(scope="module")
def hierarchical_data_10d():
    n_per_species = 10
    y_list = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 7]]
    y = np.repeat(y_list, n_per_species, axis=0).astype(float)

    # Latent 2D Hierarchy (Tree structure laid out in 2D)
    offset_0 = np.array([-50, 0]) * (1 - y[:, 0:1]) + np.array([50, 0]) * y[:, 0:1]
    offset_1 = np.array([0, -10]) * (1 - y[:, 1:2]) + np.array([0, 10]) * y[:, 1:2]
    offset_2 = np.random.randn(y.shape[0], 2)
    X_latent = offset_0 + offset_1 + offset_2

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# 5. CONTINUOUS CIRCULAR SETUP
# =============================================================================

@pytest.fixture(scope="module")
def circular_engine() -> SupervisedMDS:
    """Provides a default engine with Continuous Circular Shape."""
    return SupervisedMDS(n_components=2, manifold=CircularShape(radious=1.0))


@pytest.fixture(scope="module")
def circular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generates 10D data with a latent Continuous Circle."""
    n_points = 50

    # FIX: Use linspace instead of random.uniform
    # endpoint=False ensures we don't have 0.0 and 1.0 (which are the same point)
    y = np.linspace(0, 1, n_points, endpoint=False).astype(float)

    # Latent 2D Circle
    # y goes 0 -> 1, Angles go 0 -> 2pi
    angles = y * 2 * np.pi
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Use Helper
    return _project_and_shuffle(X_latent, y)

