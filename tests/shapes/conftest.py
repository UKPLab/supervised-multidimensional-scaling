import numpy as np
import pytest
from numpy.typing import NDArray

from smds import SupervisedMDS
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.continuous_shapes.euclidean import EuclideanShape
from smds.shapes.continuous_shapes.log_linear import LogLinearShape
from smds.shapes.continuous_shapes.semicircular import SemicircularShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape
from smds.shapes.discrete_shapes.hierarchical import HierarchicalShape
from smds.shapes.spatial_shapes.cylindrical import CylindricalShape
from smds.shapes.spatial_shapes.geodesic import GeodesicShape
from smds.shapes.spatial_shapes.spherical import SphericalShape
from smds.shapes.spiral_shape import SpiralShape


def _project_and_shuffle(
    X_latent: NDArray[np.float64], y: NDArray[np.float64], high_dim: int = 10, noise_level: float = 0.5, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Helper to project latent 2D data into high-dimensional space, add noise,
    and shuffle.
    """
    rng = np.random.default_rng(seed)
    n_samples, latent_dim = X_latent.shape

    # Create random projection matrix
    projection_matrix = rng.standard_normal((latent_dim, high_dim))

    # Project to High Dim
    X_high = X_latent @ projection_matrix

    # Add Noise
    X_high += rng.normal(scale=noise_level, size=X_high.shape)

    # Shuffle indices to ensure model doesn't rely on order
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    result: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] = (
        X_high[indices],
        y[indices],
        X_latent[indices],
    )
    return result


def _generate_lat_lon(n_samples: int, seed: int = 42) -> NDArray[np.float64]:
    """Helper to generate random (lat, lon) pairs."""
    rng = np.random.default_rng(seed)

    # Lat: -90 to 90, Lon: -180 to 180
    lat = rng.uniform(-90, 90, n_samples)
    lon = rng.uniform(-180, 180, n_samples)

    return np.stack([lat, lon], axis=1)


# =============================================================================
# CHAIN SETUP
# =============================================================================
@pytest.fixture(scope="module")
def chain_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=ChainShape(threshold=1.1))


@pytest.fixture(scope="module")
def chain_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_points = 20

    # Latent 2D Circle (Chain topology)
    y = np.arange(n_points).astype(float)
    angles = 2 * np.pi * y / n_points
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# CLUSTER SETUP
# =============================================================================
@pytest.fixture(scope="module")
def cluster_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=ClusterShape())


@pytest.fixture(scope="module")
def cluster_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    n_per_cluster = 25

    # Latent 2D Clusters
    c1 = rng.standard_normal((n_per_cluster, 2)) - 10
    c2 = rng.standard_normal((n_per_cluster, 2)) + 10
    X_latent = np.vstack([c1, c2])
    y = np.array([0.0] * n_per_cluster + [1.0] * n_per_cluster)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# DISCRETE CIRCULAR SETUP
# =============================================================================
@pytest.fixture(scope="module")
def disc_circular_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=DiscreteCircularShape(num_points=12))


@pytest.fixture(scope="module")
def disc_circular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
def hierarchical_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=HierarchicalShape(level_distances=[100.0, 10.0, 1.0]))


@pytest.fixture(scope="module")
def hierarchical_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    n_per_species = 10
    y_list = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 7]]
    y = np.repeat(y_list, n_per_species, axis=0).astype(float)

    # Latent 2D Hierarchy (Tree structure laid out in 2D)
    offset_0 = np.array([-50, 0]) * (1 - y[:, 0:1]) + np.array([50, 0]) * y[:, 0:1]
    offset_1 = np.array([0, -10]) * (1 - y[:, 1:2]) + np.array([0, 10]) * y[:, 1:2]
    offset_2 = rng.standard_normal((y.shape[0], 2))
    X_latent = offset_0 + offset_1 + offset_2

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# (CONTINUOUS) CIRCULAR SETUP
# =============================================================================


@pytest.fixture(scope="module")
def circular_engine() -> SupervisedMDS:
    """Provides a default engine with Continuous Circular Shape."""
    return SupervisedMDS(n_components=2, manifold=CircularShape(radious=1.0))


@pytest.fixture(scope="module")
def circular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generates 10D data with a latent Continuous Circle."""
    n_points = 50
    y = np.linspace(0, 1, n_points, endpoint=False).astype(float)

    # Latent 2D Circle
    angles = y * 2 * np.pi
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# CYLINDRICAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def cylindrical_engine() -> SupervisedMDS:
    # Cylinder is a 3D object
    return SupervisedMDS(n_components=3, manifold=CylindricalShape(radius=1.0))


@pytest.fixture(scope="module")
def cylindrical_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_samples = 50
    y = _generate_lat_lon(n_samples)

    # Construct Latent 3D Cylinder
    radius = 1.0
    lat_rad = np.radians(y[:, 0])
    lon_rad = np.radians(y[:, 1])

    X_latent = np.stack([radius * np.cos(lon_rad), radius * np.sin(lon_rad), lat_rad], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# SPHERICAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def spherical_engine() -> SupervisedMDS:
    # Sphere is a 3D object
    return SupervisedMDS(n_components=3, manifold=SphericalShape(radius=1.0))


@pytest.fixture(scope="module")
def spherical_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Test data for SphericalShape and GeodesicShape.

    Geodesic uses the same 3D Latent Sphere geometry as SphericalShape.
    However, the distance calculation method differs:
        - SphericalShape uses Chord distance (used here);
        - GeodesicShape uses Great Circle distance.
    """
    n_samples = 50
    y = _generate_lat_lon(n_samples)

    # Construct Latent 3D Sphere
    lat_rad = np.radians(y[:, 0])
    lon_rad = np.radians(y[:, 1])

    # Matching SphericalShape._compute_distances
    radius = 1.0
    X_latent = np.stack(
        [
            radius * np.cos(lat_rad) * np.cos(lon_rad),
            radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad),
        ],
        axis=1,
    )

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# GEODESIC SETUP
# =============================================================================
@pytest.fixture(scope="module")
def geodesic_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=3, manifold=GeodesicShape(radius=1.0))


# =============================================================================
# SPIRAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def spiral_engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=SpiralShape(initial_radius=0.5, growth_rate=1.0, num_turns=2.0))


@pytest.fixture(scope="module")
def spiral_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generates 10D data with a latent 2D Spiral structure."""
    n_samples = 50

    # Continuous labels [0, 1]
    y = np.linspace(0, 1, n_samples).astype(float)

    # Latent 2D Spiral (Matching SpiralShape logic)
    initial_radius = 0.5
    growth_rate = 1.0
    num_turns = 2.0

    theta = y * 2 * np.pi * num_turns
    radius = initial_radius + growth_rate * theta

    X_latent = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# LOG LINEAR SETUP
# =============================================================================
@pytest.fixture(scope="module")
def log_linear_engine() -> SupervisedMDS:
    # LogLinear is intrinsically 1D (a line where distance is log-scale)
    return SupervisedMDS(n_components=1, manifold=LogLinearShape(), alpha=0.1)

@pytest.fixture(scope="module")
def log_linear_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Latent 1D Line (Logarithmic spacing)
    n_points = 50
    y = np.logspace(0, 2, n_points)  # 1 to 100

    latent_1d = np.log(y + 1e-9)
    X_latent = latent_1d.reshape(-1, 1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# EUCLIDEAN SETUP
# =============================================================================
@pytest.fixture(scope="module")
def euclidean_engine() -> SupervisedMDS:
    # Euclidean maps to a 1D line
    return SupervisedMDS(n_components=1, manifold=EuclideanShape(), alpha=0.1)

@pytest.fixture(scope="module")
def euclidean_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Latent 1D Line (Linear spacing)
    n_points = 50
    y = np.linspace(0, 10, n_points).astype(float)
    X_latent = y.reshape(-1, 1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# SEMICIRCULAR SETUP
# =============================================================================
@pytest.fixture(scope="module")
def semicircular_engine() -> SupervisedMDS:
    # Semicircle is a 2D arc
    return SupervisedMDS(n_components=2, manifold=SemicircularShape(), alpha=0.1)

@pytest.fixture(scope="module")
def semicircular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Latent 2D Semicircle
    n_points = 50
    y = np.linspace(0, 1, n_points).astype(float)

    angles = y * np.pi
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return _project_and_shuffle(X_latent, y)
