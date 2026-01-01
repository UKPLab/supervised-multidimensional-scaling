import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.typing import NDArray

from smds.shapes.coordinates import CartesianCoordinates, PolarCoordinates


def test_cartesian_identity() -> None:
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    coords = CartesianCoordinates(points)
    assert coords.to_cartesian() is coords


@pytest.mark.parametrize(
    "points, expected_dists",
    [
        (np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 5.0], [5.0, 0.0]])),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
        (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0, 1.0], [1.0, 0.0, np.sqrt(2)], [1.0, np.sqrt(2), 0.0]]),
        ),
    ],
)
def test_cartesian_distances(points: NDArray[np.float64], expected_dists: NDArray[np.float64]) -> None:
    coords = CartesianCoordinates(points)
    dists = coords.compute_distances()
    assert_array_almost_equal(dists, expected_dists)


@pytest.mark.parametrize(
    "radius, theta, expected_points",
    [
        (np.array([1.0, 2.0]), np.array([0.0, np.pi / 2]), np.array([[1.0, 0.0], [0.0, 2.0]])),
        (np.array([np.sqrt(2)]), np.array([np.pi / 4]), np.array([[1.0, 1.0]])),
        (np.array([0.0]), np.array([0.0]), np.array([[0.0, 0.0]])),
    ],
)
def test_polar_to_cartesian(
    radius: NDArray[np.float64], theta: NDArray[np.float64], expected_points: NDArray[np.float64]
) -> None:
    polar = PolarCoordinates(radius, theta)
    cart = polar.to_cartesian()
    assert_array_almost_equal(cart.points, expected_points)
