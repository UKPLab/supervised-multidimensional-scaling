import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class ClusterShape(BaseShape):
    """
    Compute ideal distances for categorical data (0 for same, 1 for different).

    This shape models data where the only meaningful distinction is category
    membership. The ideal distance is defined as 0 for points within the same
    category and 1 for points in different categories.

    Parameters
    ----------
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects a 1D array of category labels.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = False):
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the binary pairwise distance matrix.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of categorical labels.

        Returns
        -------
        NDArray[np.float64]
            A pairwise distance matrix where $D_{ij} = 0$ if $y_i = y_j$ and
            $D_{ij} = 1$ otherwise.
        """
        distance_matrix: NDArray[np.float64] = (y[:, None] != y[None, :]).astype(float)

        return distance_matrix
