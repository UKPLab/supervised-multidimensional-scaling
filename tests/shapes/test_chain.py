import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from smds.shapes.discrete_shapes import ChainShape


def test_chain_init_validation() -> None:
    """Tests that the __init__ method validates its parameters correctly."""
    with pytest.raises(ValueError, match="threshold must be positive"):
        ChainShape(threshold=0)


def test_chain_distance_computation() -> None:
    """Tests the core neighbor-only wrap-around distance logic."""
    shape = ChainShape(threshold=2.0)
    y: NDArray[np.float64] = np.array([0, 1, 2, 3]).astype(float)
    dists: NDArray[np.float64] = shape(y)
    expected: NDArray[np.float64] = np.array(
        [
            [0.0, 1.0, -1.0, 1.0],
            [1.0, 0.0, 1.0, -1.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, 0.0],
        ]
    )
    assert_array_almost_equal(dists, expected)
