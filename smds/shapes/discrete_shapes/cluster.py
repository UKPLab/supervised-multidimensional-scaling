import numpy as np

from smds.shapes.base_shape import BaseShape


class ClusterShape(BaseShape):
    """
        Implements the cluster-shape for categorical data.

        This shape models data where the only meaningful distinction is category
        membership. The ideal distance is 0 for points within the same category
        and 1 for points in different categories.
        """
    def __init__(self) -> None:
        pass

    def _compute_distances(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the ideal pairwise distance matrix for categorical labels.

        Args:
            y: A 1D numpy array of labels of shape (n_samples,).

        Returns:
            A (n_samples, n_samples) distance matrix where D[i, j] is 0 if
            y[i] == y[j] and 1 otherwise.
        """
        distance_matrix = (y[:, None] != y[None, :]).astype(float)

        return distance_matrix

