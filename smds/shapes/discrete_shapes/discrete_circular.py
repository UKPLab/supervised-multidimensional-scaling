from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class DiscreteCircularShape(BaseShape):
    """
    Implements the discrete circular shape for ordered, cyclical data.

    This shape is designed for features with a fixed number of ordered steps
    that "wrap around," such as months of the year, days of the week, or
    hours on a clock. The resulting projection should form a ring or polygon
    where adjacent categories are placed next to each other.
    """

    def __init__(self, num_points: Optional[int] = None) -> None:
        """
        Initialize the DiscreteCircularShape.

        Args:
            num_points (Optional[int]): The total number of points on the circle.
                                        For a 12-hour clock with labels 0-11,
                                        this would be 12. If None (default), it will
                                        be inferred from the max value in the data
                                        (max(y) + 1), which is less robust.
        """
        if num_points is not None and num_points <= 0:
            raise ValueError("num_points must be a positive integer.")
        self.num_points = num_points

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate that y is a 1D array of numeric labels.
        This follows the pattern from HierarchicalShape but is adapted for 1D data.
        """
        # NOTE: This still enforces float64 as per the BaseShape contract.
        # For a "discrete" shape, one might expect integers.
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 1:
            raise ValueError(
                f"Input 'y' for DiscreteCircularShape must be 1-dimensional (n_samples,), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )

        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
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
                distance_matrix[i, j] = min(abs(y[i] - y[j]), max_y + 1 - abs(y[i] - y[j]))

        return distance_matrix
