import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape
from smds.shapes.coordinates.polar_coordinates import PolarCoordinates


class SpiralShape(BaseShape):
    """Arrange points in an Archimedean spiral pattern.
    initial_radius: Starting radius of the spiral
    growth_rate: Rate at which spiral expands per radian
    num_turns: Number of complete rotations
    """
    def __init__(self, initial_radius: float = 0.5, growth_rate: float = 1.0, num_turns: float = 2.0) -> None:
        self.initial_radius = initial_radius
        self.growth_rate = growth_rate
        self.num_turns = num_turns

    def _normalize_y(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_range = np.ptp(y)
        if y_range == 0:
            return np.zeros_like(y)
        return (y - y.min()) / y_range

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm = self._normalize_y(y)
        theta = y_norm * 2 * np.pi * self.num_turns
        radius = self.initial_radius + self.growth_rate * theta

        polar = PolarCoordinates(radius, theta)
        cartesian = polar.to_cartesian()

        return cartesian.compute_distances()
