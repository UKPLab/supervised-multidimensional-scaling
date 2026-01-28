from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class DiscreteCircularShape(BaseShape):
    """
    Compute distances for ordered, cyclical data (e.g., months, hours).

    This shape models features with a fixed number of ordered steps that wrap
    around (periodic boundary conditions). The ideal geometry forms a ring or
    regular polygon where adjacent integer categories are equidistant.

    Parameters
    ----------
    num_points : int, optional
        The total cycle length (modulus). For example, 12 for months or 24 for hours.
        If None, it is inferred as `max(y) + 1`.
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False,
        as discrete shapes usually rely on raw integer steps.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects 1D array of discrete steps.
    """

    y_ndim = 1
    # NOTE: This still enforces float64 as per the BaseShape contract.
    # For a "discrete" shape, one might expect integers.

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, num_points: Optional[int] = None, normalize_labels: bool = False) -> None:
        if num_points is not None and num_points <= 0:
            raise ValueError("num_points must be a positive integer.")
        self.num_points = num_points
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Compute the shortest ring distance (circular arc length) between points.

        Calculates the minimum distance along the cycle:
        $$ D_{ij} = \min(|y_i - y_j|, C - |y_i - y_j|) $$
        where $C$ is the cycle length.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of labels representing steps on the circle.

        Returns
        -------
        NDArray[np.float64]
            Pairwise distance matrix representing the shortest path on the ring.
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
