import numpy as np

from smds.shapes.base_shape import BaseShape


class DiscreteCircularShape(BaseShape):
    """
    Implements the discrete circular shape for ordered, cyclical data.

    This shape is designed for features with a fixed number of ordered steps
    that "wrap around," such as months of the year, days of the week, or
    hours on a clock. The resulting projection should form a ring or polygon
    where adjacent categories are placed next to each other.
    """
    def __init__(self) -> None:
        pass

    def _compute_distances(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the ideal pairwise distance matrix for discrete circular labels.

        Args:
            y: A 1D numpy array of labels of shape (n_samples,).

        Returns:
            A (n_samples, n_samples) distance matrix representing the shortest
            ring distance between each pair of labels.
        """
        n = len(y)
        max_y = np.max(y)

        distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = min(
                    abs(y[i] - y[j]),
                    max_y + 1 - abs(y[i] - y[j])
                )

        return distance_matrix
