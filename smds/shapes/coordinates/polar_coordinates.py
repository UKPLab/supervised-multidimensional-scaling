import numpy as np
from numpy.typing import NDArray

from smds.shapes.coordinates import BaseCoordinates, CartesianCoordinates


class PolarCoordinates(BaseCoordinates):
    def __init__(self, radius: NDArray[np.float64], theta: NDArray[np.float64]) -> None:
        self.radius = radius
        self.theta = theta

    def to_cartesian(self) -> CartesianCoordinates:
        x = self.radius * np.cos(self.theta)
        y = self.radius * np.sin(self.theta)
        return CartesianCoordinates(np.column_stack([x, y]))
