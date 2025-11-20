from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from smds.shapes.coordinates.base_coordinates import BaseCoordinates


class CartesianCoordinates(BaseCoordinates):
    def __init__(self, points: NDArray[np.float64]) -> None:
        self.points = points

    def to_cartesian(self) -> CartesianCoordinates:
        return self

    def compute_distances(self) -> NDArray[np.float64]:
        return squareform(pdist(self.points, metric="euclidean"))
