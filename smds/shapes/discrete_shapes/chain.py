import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class ChainShape(BaseShape):
    """
    Implements a cyclical chain shape for ordered, sequential data.

    This shape models a sequence of points in a closed loop where only adjacent
    items have a defined distance. Distances between non-neighbors are marked as
    undefined (-1.0).
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, threshold: float = 2.0, normalize_labels: bool = False):
        """
        Initialize the ChainShape.

        Args:
            threshold (float): The distance below which points are considered neighbors.
                               The original default was 2.0, which connects points
                               with an integer distance of 1.
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes a sparse distance matrix where only neighbors in a cycle are connected.
        """
        cycle_length = np.max(y) + 1

        direct_dist = np.abs(y[:, None] - y[None, :])
        wrap_around_dist = cycle_length - direct_dist
        base_distances = np.minimum(direct_dist, wrap_around_dist)

        distance_matrix = np.where(base_distances < self.threshold, base_distances, -1.0)

        return distance_matrix
