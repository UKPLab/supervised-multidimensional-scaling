import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class ClusterShape(BaseShape):
    """
    Implements the cluster-shape for categorical data.

    This shape models data where the only meaningful distinction is category
    membership. The ideal distance is 0 for points within the same category
    and 1 for points in different categories.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = False):
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the ideal pairwise distance matrix for categorical labels.

        Args:
            y: A 1D numpy array of labels of shape (n_samples,).

        Returns
        -------
            A (n_samples, n_samples) distance matrix where D[i, j] is 0 if
            y[i] == y[j] and 1 otherwise.
        """
        distance_matrix: NDArray[np.float64] = (y[:, None] != y[None, :]).astype(float)

        return distance_matrix
