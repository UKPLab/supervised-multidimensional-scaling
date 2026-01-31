import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class ChainShape(BaseShape):
    """
    Compute sparse cyclic distances where non-neighbors are disconnected.

    This shape models a closed loop sequence. Unlike `DiscreteCircularShape`,
    which computes the full distance matrix, this shape enforces locality:
    points separated by a distance greater than or equal to `threshold` are
    marked as disconnected (distance = -1.0).

    Parameters
    ----------
    threshold : float, optional
        The distance cutoff for defining neighbors. Pairs with a cyclic distance
        less than this value are connected. Default is 2.0 (connects adjacent
        integers with distance 1).
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects ordered sequential data.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, threshold: float = 2.0, normalize_labels: bool = False):
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the thresholded cyclic distance matrix.

        Calculates the shortest ring distance $d_{cycle}$. If $d_{cycle} < \\text{threshold}$,
        the distance is preserved; otherwise, it is set to -1.0 to indicate no connection.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of ordered labels.

        Returns
        -------
        NDArray[np.float64]
            A pairwise distance matrix where disconnected pairs have value -1.0.
        """
        cycle_length = np.max(y) + 1

        direct_dist = np.abs(y[:, None] - y[None, :])
        wrap_around_dist = cycle_length - direct_dist
        base_distances = np.minimum(direct_dist, wrap_around_dist)

        distance_matrix = np.where(base_distances < self.threshold, base_distances, -1.0)

        return distance_matrix
