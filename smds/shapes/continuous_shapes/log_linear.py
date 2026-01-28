import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class LogLinearShape(BaseShape):
    """
    Compute distances based on logarithmic scaling.

    This shape models data where differences are more significant at smaller scales
    than at larger scales (e.g., sound intensity, earthquake magnitude).
    The distance is defined as the absolute difference between the logarithms
    of the values.

    Reference: Table 1 in the "Shape Happens" paper.

    Parameters
    ----------
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects non-negative continuous values.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = False):
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input is 1D and contains only non-negative values.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of values.

        Returns
        -------
        NDArray[np.float64]
            Validated flat array.

        Raises
        ------
        ValueError
            If `y` is empty, multi-dimensional, or contains negative values.
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
        """
        Compute pairwise logarithmic distances.

        Calculates $D_{ij} = |\\log(y_i + 1) - \\log(y_j + 1)|$.
        A shift of 1.0 is added to avoid $\\log(0)$.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of non-negative values.

        Returns
        -------
        NDArray[np.float64]
            Pairwise distance matrix representing the difference in magnitude.
        """
        y_flat = y.ravel()
        y_log = np.log(y_flat + 1.0)
        distance_matrix: NDArray[np.float64] = np.abs(y_log[:, None] - y_log[None, :])

        return distance_matrix
