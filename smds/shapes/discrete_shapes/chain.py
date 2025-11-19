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

    def __init__(self, threshold: float = 2.0) -> None:
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

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Validate that y is a 1D array of numeric labels."""
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 1:
            raise ValueError(
                f"Input 'y' for ChainShape must be 1-dimensional (n_samples,), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )
        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes a sparse distance matrix where only neighbors in a cycle are connected.

        cycle_length = np.max(y) + 1

        direct_dist = np.abs(y[:, None] - y[None, :])
        wrap_around_dist = cycle_length - direct_dist
        base_distances = np.minimum(direct_dist, wrap_around_dist)

        distance_matrix = np.where(
            base_distances < self.threshold,
            base_distances,
            -1.0
        )
        """
        n = len(y)
        max_y = np.max(y)
        cycle_length = max_y + 1

        distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                direct_dist = np.abs(y[i] - y[j])
                wrap_around_dist = cycle_length - direct_dist
                dist = min(direct_dist, wrap_around_dist)

                # Apply the threshold
                distance_matrix[i, j] = dist if dist < self.threshold else -1.0

        return distance_matrix
