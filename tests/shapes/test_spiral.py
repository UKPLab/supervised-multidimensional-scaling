import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from smds.shapes.spiral_shape import SpiralShape


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
    normalized = spiral._normalize_y(y_input)
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
    verify degenerate edge cases where the spiral parameters force all input points to map to the exact same physical location
    """
    shape = SpiralShape(initial_radius=init_radius, growth_rate=growth, num_turns=turns)
    dists = shape._compute_distances(y_input)
    assert_array_almost_equal(np.sum(dists), expected_sum)
