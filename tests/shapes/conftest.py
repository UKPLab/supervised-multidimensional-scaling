import numpy as np
import pytest
from numpy.typing import NDArray

from smds import SupervisedMDS


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
def chain_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="chain")


@pytest.fixture(scope="module")
def chain_engine_user_provided_stage1(
    chain_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="chain")


@pytest.fixture(scope="module")
def chain_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Note: Even though this is called a "chain", the data also forms a discrete circle
    # and which is also a line (eucledian).
    # Truly chain-specific data (where the chain score is highest compared to all other
    # shapes) may need to be found separately, if it exists at all.
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
def cluster_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="cluster")


@pytest.fixture(scope="module")
def cluster_engine_user_provided_stage1(
    cluster_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="cluster")


@pytest.fixture(scope="module")
def cluster_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    n = 100
    # 3 Clusters
    c1 = rng.standard_normal((n, 2)) + [0, 10]
    c2 = rng.standard_normal((n, 2)) + [-8.66, -5]
    c3 = rng.standard_normal((n, 2)) + [8.66, -5]

    X_latent = np.vstack([c1, c2, c3])
    y = np.array([0.0] * n + [1.0] * n + [2.0] * n)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# DISCRETE CIRCULAR SETUP
# =============================================================================
@pytest.fixture(scope="module")
def disc_circular_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="discrete_circular")


@pytest.fixture(scope="module")
def disc_circular_engine_user_provided_stage1(
    disc_circular_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="discrete_circular")


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
def hierarchical_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="hierarchical")


@pytest.fixture(scope="module")
def hierarchical_engine_user_provided_stage1(
    hierarchical_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="hierarchical")


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
def circular_engine_computed_stage1() -> SupervisedMDS:
    """Provides a default engine with Continuous Circular Shape."""
    return SupervisedMDS(stage_1="computed", manifold="circular")


@pytest.fixture(scope="module")
def circular_engine_user_provided_stage1(
    circular_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    """Provides a default engine with Continuous Circular Shape."""
    return SupervisedMDS(stage_1="user_provided", manifold="circular")


@pytest.fixture(scope="module")
def circular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generates 10D data with a latent Continuous Circle."""
    n_points = 100
    y = np.linspace(0, 1, n_points, endpoint=False).astype(float)

    # Latent 2D Circle
    angles = y * 2 * np.pi
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# CYLINDRICAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def cylindrical_engine_computed_stage1() -> SupervisedMDS:
    # Cylinder is a 3D object
    return SupervisedMDS(stage_1="computed", manifold="cylindrical", radius=1.0)


@pytest.fixture(scope="module")
def cylindrical_engine_user_provided_stage1(
    cylindrical_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    # Cylinder is a 3D object
    return SupervisedMDS(stage_1="user_provided", manifold="cylindrical", radius=1.0)


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
def spherical_engine_computed_stage1() -> SupervisedMDS:
    # Sphere is a 3D object
    return SupervisedMDS(stage_1="computed", manifold="spherical", radius=1.0)


@pytest.fixture(scope="module")
def spherical_engine_user_provided_stage1(
    spherical_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    # Sphere is a 3D object
    return SupervisedMDS(stage_1="user_provided", manifold="spherical", radius=1.0)


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
def geodesic_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="geodesic", radius=1.0)


@pytest.fixture(scope="module")
def geodesic_engine_user_provided_stage1(
    spherical_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="geodesic", radius=1.0)


# =============================================================================
# SPIRAL SETUP
# =============================================================================
@pytest.fixture(scope="module")
def spiral_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="spiral")


@pytest.fixture(scope="module")
def spiral_engine_user_provided_stage1(
    spiral_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="spiral")


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
def log_linear_engine_computed_stage1() -> SupervisedMDS:
    # LogLinear is intrinsically 1D (a line where distance is log-scale)
    return SupervisedMDS(stage_1="computed", manifold="log_linear", alpha=0.1)


@pytest.fixture(scope="module")
def log_linear_engine_user_provided_stage1(
    log_linear_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    # LogLinear is intrinsically 1D (a line where distance is log-scale)
    return SupervisedMDS(stage_1="user_provided", manifold="log_linear", alpha=0.1)


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
def euclidean_engine_computed_stage1() -> SupervisedMDS:
    # Euclidean maps to a 1D line
    return SupervisedMDS(stage_1="computed", manifold="euclidean", alpha=0.1)


@pytest.fixture(scope="module")
def euclidean_engine_user_provided_stage1(
    euclidean_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    # Euclidean maps to a 1D line
    return SupervisedMDS(stage_1="user_provided", manifold="euclidean", alpha=0.1)


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
def semicircular_engine_computed_stage1() -> SupervisedMDS:
    # Semicircle is a 2D arc
    return SupervisedMDS(stage_1="computed", manifold="semicircular", alpha=0.1)


@pytest.fixture(scope="module")
def semicircular_engine_user_provided_stage1(
    semicircular_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    # Semicircle is a 2D arc
    return SupervisedMDS(stage_1="user_provided", manifold="semicircular", alpha=0.1)


@pytest.fixture(scope="module")
def semicircular_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Latent 2D Semicircle
    n_points = 50
    y = np.linspace(0, 1, n_points).astype(float)

    angles = y * np.pi
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return _project_and_shuffle(X_latent, y)


# =============================================================================
# KLEIN BOTTLE SETUP
# =============================================================================
@pytest.fixture(scope="module")
def klein_bottle_engine_computed_stage1() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="klein_bottle", alpha=0.1)


@pytest.fixture(scope="module")
def klein_bottle_engine_user_provided_stage1(
    klein_bottle_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> SupervisedMDS:
    return SupervisedMDS(stage_1="user_provided", manifold="klein_bottle", alpha=0.1)


@pytest.fixture(scope="module")
def klein_bottle_data_10d() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates 10D data with a latent 4D Klein Bottle structure.
    Using 4D ensures the topology is mathematically valid (intersection-free).
    """
    n_samples = 50
    rng = np.random.default_rng(42)

    u_raw = rng.uniform(0, 1, n_samples)
    v_raw = rng.uniform(0, 1, n_samples)
    y = np.stack([u_raw, v_raw], axis=1)

    u = u_raw * 2 * np.pi
    v = v_raw * 2 * np.pi

    R = 3
    P = 1

    x1 = (R + P * np.cos(v)) * np.cos(u)
    x2 = (R + P * np.cos(v)) * np.sin(u)
    x3 = P * np.sin(v) * np.cos(u / 2)
    x4 = P * np.sin(v) * np.sin(u / 2)

    X_latent = np.stack([x1, x2, x3, x4], axis=1)

    return _project_and_shuffle(X_latent, y)
