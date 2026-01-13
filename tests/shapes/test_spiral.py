import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from smds import ComputedSMDSParametrization, SupervisedMDS
from smds.shapes.continuous_shapes import SpiralShape


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    return SupervisedMDS(stage_1=ComputedSMDSParametrization(n_components=2, manifold=SpiralShape()))


@pytest.fixture
def random_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    X: NDArray[np.float64] = rng.standard_normal((50, 10))
    y: NDArray[np.float64] = rng.standard_normal((50,))  # Continuous values for spiral
    return X, y


@pytest.fixture
def spiral() -> SpiralShape:
    return SpiralShape(initial_radius=0.5, growth_rate=1.0, num_turns=1.0)


@pytest.mark.parametrize("init_radius, growth, turns", [(0.5, 1.0, 1.0), (0.0, 100.0, 5.0), (1.0, 0.0, 0.0)])
def test_initialization_params(init_radius: float, growth: float, turns: float) -> None:
    spiral = SpiralShape(initial_radius=init_radius, growth_rate=growth, num_turns=turns)
    assert spiral.initial_radius == init_radius
    assert spiral.growth_rate == growth
    assert spiral.num_turns == turns


@pytest.mark.parametrize(
    "y_input, expected_output",
    [
        (np.array([10.0, 15.0, 20.0]), np.array([0.0, 0.5, 1.0])),
        (np.array([5.0, 5.0, 5.0]), np.zeros(3)),
        (np.array([-10.0, 0.0, 10.0]), np.array([0.0, 0.5, 1.0])),
        (np.array([42.0]), np.array([0.0])),
    ],
    ids=["standard_range", "constant_array", "negative_values", "single_value"],
)
def test_normalize_y(spiral: SpiralShape, y_input: NDArray[np.float64], expected_output: NDArray[np.float64]) -> None:
    normalized = spiral._do_normalize_labels(y_input)
    assert_array_almost_equal(normalized, expected_output)


def test_compute_distances_structure(spiral: SpiralShape) -> None:
    y = np.array([0.0, 0.5, 1.0])
    dists = spiral._compute_distances(y)

    assert dists.shape == (3, 3)
    assert_array_almost_equal(np.diag(dists), np.zeros(3))
    assert_array_almost_equal(dists, dists.T)
    assert np.all(dists >= 0)


@pytest.mark.parametrize(
    "init_radius, growth, turns, y_input, expected_sum",
    [
        (1.0, 0.0, 0.0, np.array([0.0, 1.0]), 0.0),
        (1.0, 5.0, 0.0, np.array([0.0, 1.0]), 0.0),
    ],
)
def test_compute_distances_values(
    init_radius: float, growth: float, turns: float, y_input: NDArray[np.float64], expected_sum: float
) -> None:
    """
    Verify degenerate edge cases where the spiral parameters force all input points to map to the exact same physical
     location
    """
    shape = SpiralShape(initial_radius=init_radius, growth_rate=growth, num_turns=turns)
    dists = shape._compute_distances(y_input)
    assert_array_almost_equal(np.sum(dists), expected_sum)


@pytest.mark.smoke
def test_spiral_smoke_test(
    random_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    A simple "smoke test" to ensure SpiralShape can be fit and transformed
    without errors and produces an output of the correct shape.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = random_data

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    n_samples = X.shape[0]
    n_components = smds_engine.stage_1.n_components

    assert X_proj.shape == (n_samples, n_components), (
        f"Output shape is incorrect. "
        f"Expected (n_samples, n_components): {(n_samples, n_components)}, "
        f"but got: {X_proj.shape}."
    )


@pytest.fixture
def structured_spiral_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a dataset where points are already arranged in a spiral pattern.
    """
    n_samples = 50
    y: NDArray[np.float64] = np.linspace(0, 1, n_samples)
    initial_radius = 0.5
    growth_rate = 1.0
    num_turns = 2.0
    theta = y * 2 * np.pi * num_turns
    radius = initial_radius + growth_rate * theta

    X: NDArray[np.float64] = np.column_stack(
        [
            radius * np.cos(theta) + np.random.randn(n_samples) * 0.1,
            radius * np.sin(theta) + np.random.randn(n_samples) * 0.1,
        ]
    )

    indices: NDArray[np.int_] = np.arange(n_samples)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_spiral_preserves_order_in_2d(
    structured_spiral_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Sanity Check (2D -> 2D): Tests that SpiralShape preserves the ordering
    along the spiral path - points with nearby y values should be nearby in the projection.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = structured_spiral_data_2d

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    sort_indices = np.argsort(y)
    y_sorted = y[sort_indices]
    X_proj_sorted = X_proj[sort_indices]

    path_distances = np.linalg.norm(np.diff(X_proj_sorted, axis=0), axis=1)

    y_differences = np.diff(y_sorted)

    correlation, p_value = spearmanr(y_differences, path_distances)

    correlation_threshold = 0.3
    assert correlation > correlation_threshold, (
        f"Spiral ordering is not preserved. "
        f"Spearman correlation between y-differences and path distances ({correlation:.4f}) "
        f"should be greater than {correlation_threshold}."
    )


@pytest.fixture
def structured_spiral_data_high_dim() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a high-dimensional dataset containing a hidden 2D spiral structure.
    This tests the ability of SMDS to recover a latent spiral manifold.
    """
    latent_dim = 2
    high_dim = 10
    n_samples = 50

    # Create the ground-truth spiral in 2D
    y: NDArray[np.float64] = np.linspace(0, 1, n_samples)
    initial_radius = 0.5
    growth_rate = 1.0
    num_turns = 2.0
    theta = y * 2 * np.pi * num_turns
    radius = initial_radius + growth_rate * theta

    X_latent: NDArray[np.float64] = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    projection_matrix: NDArray[np.float64] = np.random.randn(latent_dim, high_dim)
    X_high_dim: NDArray[np.float64] = X_latent @ projection_matrix

    indices: NDArray[np.int_] = np.arange(n_samples)
    np.random.shuffle(indices)

    return X_high_dim[indices], y[indices], X_latent[indices]
