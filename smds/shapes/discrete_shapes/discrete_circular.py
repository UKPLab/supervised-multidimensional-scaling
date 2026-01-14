from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class DiscreteCircularShape(BaseShape):
    """
    Implements the discrete circular shape for ordered, cyclical data.

    This shape is designed for features with a fixed number of ordered steps
    that "wrap around," such as months of the year, days of the week, or
    hours on a clock. The resulting projection should form a ring or polygon
    where adjacent categories are placed next to each other.
    """

    y_ndim = 1
    # NOTE: This still enforces float64 as per the BaseShape contract.
    # For a "discrete" shape, one might expect integers.

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, num_points: Optional[int] = None, normalize_labels: bool = False) -> None:
        """
        Initialize the DiscreteCircularShape.

        Args:
            num_points (Optional[int]): The total number of points on the circle.
                                        For a 12-hour clock (0-11), this is 12.
                                        If None, it is inferred as max(y) + 1.
            normalize_labels (bool):    Defaults to False. Discrete shapes rely on
                                        integer steps and should usually not be normalized.
        """
        if num_points is not None and num_points <= 0:
            raise ValueError("num_points must be a positive integer.")
        self.num_points = num_points
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the ideal pairwise distance matrix for discrete circular labels.

        Args:
            y: A 1D numpy array of labels of shape (n_samples,).

        Returns
        -------
            A (n_samples, n_samples) distance matrix representing the shortest
            ring distance between each pair of labels.
        """
        # Determine cycle length: Use self.num_points if available, else infer.
        if self.num_points is not None:
            cycle_length = float(self.num_points)
        else:
            cycle_length = np.max(y) + 1.0

        # Direct absolute difference
        direct_dist = np.abs(y[:, None] - y[None, :])

        # Reduce modulo cycle_length to handle labels outside [0, cycle_length-1]
        direct_dist = np.mod(direct_dist, cycle_length)

        # Wrap-around difference
        wrap_around_dist = cycle_length - direct_dist

        # Shortest path on the ring
        distance_matrix: NDArray[np.float64] = np.minimum(direct_dist, wrap_around_dist)

        return distance_matrix
