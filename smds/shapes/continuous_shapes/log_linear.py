from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class LogLinearShape(BaseShape):
    """
    Implements the Log-Linear shape for continuous data.

    This models data where distance scales logarithmically.
    Reference: Table 1 in "Shape Happens" paper.
    Formula: d(i, j) = |log(yi) - log(yj)|
    """

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, normalize_labels: Optional[bool] = False): # Removed epsilon
        """
        Args:
            normalize_labels: Whether to scale inputs to [0, 1].
        """
        self._normalize_labels = normalize_labels if normalize_labels is not None else False

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validates that input y is a non-empty 1D array of non-negative values.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim > 1 and y_proc.shape[1] > 1:
            raise ValueError(
                f"Input 'y' for LogLinearShape must be 1-dimensional (n_samples,) "
                f"or (n_samples, 1), but got shape {y_proc.shape}."
            )

        y_flat = y_proc.ravel()

        if np.any(y_flat < 0):
             raise ValueError("Input 'y' for LogLinearShape cannot contain negative values.")

        return y_flat

def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_flat = y.ravel()
        y_log = np.log(y_flat + 1.0) 
        distance_matrix: NDArray[np.float64] = np.abs(y_log[:, None] - y_log[None, :])

        return distance_matrix
