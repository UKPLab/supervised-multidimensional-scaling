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

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(
        self,
        initial_radius: float = 0.5,
        growth_rate: float = 1.0,
        num_turns: float = 2.0,
        normalize_labels: bool = True,
    ) -> None:
        self.initial_radius = initial_radius
        self.growth_rate = growth_rate
        self.num_turns = num_turns
        self._normalize_labels = normalize_labels

    @staticmethod
    def _do_normalize_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_range = np.ptp(y)
        if y_range == 0:
            zero_array: NDArray[np.float64] = np.zeros_like(y)
            return zero_array
        result: NDArray[np.float64] = (y - y.min()) / y_range
        return result

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = y * 2 * np.pi * self.num_turns
        radius = self.initial_radius + self.growth_rate * theta

        polar = PolarCoordinates(radius, theta)
        cartesian = polar.to_cartesian()

        result: NDArray[np.float64] = cartesian.compute_distances()
        return result
