import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class EuclideanShape(BaseShape):
    """
    Implements the Euclidean (Linear) shape for continuous data.

    Reference: Table 1 in "Shape Happens" paper (referred to as 'linear').
    Formula: d(i, j) = |yi - yj|
    """

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = True):
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validates that input y is a non-empty 1D array.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim > 1 and y_proc.shape[1] > 1:
            raise ValueError(
                f"Input 'y' for EuclideanShape must be 1-dimensional (n_samples,) "
                f"or (n_samples, 1), but got shape {y_proc.shape}."
            )

        return y_proc.ravel()

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_flat = y.ravel()

        distance_matrix: NDArray[np.float64] = np.abs(y_flat[:, None] - y_flat[None, :])

        return distance_matrix
