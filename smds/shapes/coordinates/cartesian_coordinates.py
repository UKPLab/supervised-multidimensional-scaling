from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

from smds.shapes.coordinates import BaseCoordinates


class CartesianCoordinates(BaseCoordinates):
    """
    Represent points in Cartesian space and compute Euclidean distances.

    Parameters
    ----------
    points : NDArray[np.float64]
        An array of shape (n_samples, n_dimensions) representing the coordinates
        of points in N-dimensional Cartesian space.
    """

    def __init__(self, points: NDArray[np.float64]) -> None:
        self.points = points

    def to_cartesian(self) -> CartesianCoordinates:
        """
        Return self, as the coordinates are already Cartesian.

        Returns
        -------
        CartesianCoordinates
            The current instance.
        """
        return self

    def compute_distances(self) -> NDArray[np.float64]:
        """
        Compute the pairwise Euclidean distance matrix for the stored points.

        Calculates $D_{ij} = ||x_i - x_j||_2$ using efficient vectorized operations.

        Returns
        -------
        NDArray[np.float64]
            A square, symmetric distance matrix of shape (n_samples, n_samples)
            with zeros on the diagonal.
        """
        result: NDArray[np.float64] = squareform(pdist(self.points, metric="euclidean"))
        return result
