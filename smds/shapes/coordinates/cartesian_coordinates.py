from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from smds.shapes.coordinates.base_coordinates import BaseCoordinates


class CartesianCoordinates(BaseCoordinates):
    def __init__(self, points: NDArray[np.float64]) -> None:
        self.points = points

    def to_cartesian(self) -> CartesianCoordinates:
        return self

    def compute_distances(self) -> NDArray[np.float64]:
        coords_sq = np.sum(self.points**2, axis=1)
        return np.sqrt(coords_sq[:, None] + coords_sq[None, :] - 2 * self.points @ self.points.T)
