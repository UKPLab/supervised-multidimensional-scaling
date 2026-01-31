import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from smds.shapes.discrete_shapes import DiscreteCircularShape


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
