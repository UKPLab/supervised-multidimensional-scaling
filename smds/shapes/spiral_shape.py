from typing import Optional

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

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self,
                 initial_radius: Optional[float] = 0.5,
                 growth_rate: Optional[float] = 1.0,
                 num_turns: Optional[float] = 2.0,
                 normalize_labels: Optional[bool] = True) -> None:

        self.initial_radius: float = initial_radius if initial_radius is not None else 0.5
        self.growth_rate: float = growth_rate if growth_rate is not None else 1.0
        self.num_turns: float = num_turns if num_turns is not None else 2.0
        self._normalize_labels: bool = bool(normalize_labels) if normalize_labels is not None else True

    @staticmethod
    def _do_normalize_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_range = np.ptp(y)
        if y_range == 0:
            result: NDArray[np.float64] = np.zeros_like(y)
            return result
        result = (y - y.min()) / y_range
        return result

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        theta = y * 2 * np.pi * self.num_turns
        radius = self.initial_radius + self.growth_rate * theta

        polar = PolarCoordinates(radius, theta)
        cartesian = polar.to_cartesian()

        result: NDArray[np.float64] = cartesian.compute_distances()
        return result
