import numpy as np
from numpy.typing import NDArray

from smds.shapes.coordinates import BaseCoordinates, CartesianCoordinates


class PolarCoordinates(BaseCoordinates):
    """
    Represent points in Polar coordinates ($r, \theta$).

    Parameters
    ----------
    radius : NDArray[np.float64]
        Radial distance from the origin.
    theta : NDArray[np.float64]
        Azimuthal angle in radians.
    """

    def __init__(self, radius: NDArray[np.float64], theta: NDArray[np.float64]) -> None:
        self.radius = radius
        self.theta = theta

    def to_cartesian(self) -> CartesianCoordinates:
        """
        Convert polar coordinates to 2D Cartesian coordinates.

        The transformation is defined as:
        $$ x = r \\cos(\\theta) $$
        $$ y = r \\sin(\\theta) $$

        Returns
        -------
        CartesianCoordinates
            The resulting points in Cartesian space.
        """
        x = self.radius * np.cos(self.theta)
        y = self.radius * np.sin(self.theta)
        return CartesianCoordinates(np.column_stack([x, y]))
