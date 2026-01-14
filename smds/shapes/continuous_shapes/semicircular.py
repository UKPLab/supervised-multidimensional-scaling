import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class SemicircularShape(BaseShape):
    r"""
    Compute Euclidean (chord) distances for points mapped to a semicircle.

    This shape maps normalized 1D scalar values to angles on a unit semicircle
    (ranging from 0 to $\pi$) and computes the straight-line (chord) distance
    between them.

    Parameters
    ----------
    normalize_labels : bool, optional
        Whether to normalize labels to the range [0, 1]. Default is True.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects 1D scalar values.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = True):
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input is 1D.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array.

        Returns
        -------
        NDArray[np.float64]
            Validated flat 1D array.

        Raises
        ------
        ValueError
            If `y` is empty or has more than one column.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim > 1 and y_proc.shape[1] > 1:
            raise ValueError(
                f"Input 'y' for SemicircularShape must be 1-dimensional (n_samples,) "
                f"or (n_samples, 1), but got shape {y_proc.shape}."
            )

        return y_proc.ravel()

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute pairwise chord distances on the unit semicircle.

        The distance is calculated as:
        $$ D_{ij} = 2 \\sin\\left(\\frac{\\pi}{2} |y_i - y_j|\\right) $$
        This corresponds to mapping $y$ (assumed in $[0, 1]$) to angles $\\theta \\in [0, \\pi]$
        and finding the chord length.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of labels (typically normalized).

        Returns
        -------
        NDArray[np.float64]
            Pairwise Euclidean distance matrix.
        """
        y_flat = y.ravel()

        delta: NDArray[np.float64] = np.abs(y_flat[:, None] - y_flat[None, :])

        distance_matrix: NDArray[np.float64] = 2 * np.sin((np.pi / 2) * delta)

        return distance_matrix
