import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class HierarchicalShape(BaseShape):
    """
    Compute distances based on hierarchical (tree-structured) categorical data.

    This shape models data organized in levels (e.g., Country > State > City).
    The distance between two points is determined by the specific level at which
    they first diverge. Higher levels (earlier indices) typically represent
    larger conceptual distances.



    Parameters
    ----------
    level_distances : NDArray[np.float64]
        An array of distance penalties corresponding to each level of the hierarchy.
        `level_distances[0]` is the distance applied if points differ at the
        root level (column 0). `level_distances[i]` is used if points match
        up to level `i-1` but differ at level `i`.
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False,
        as hierarchical labels are typically discrete categories.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (2). Expects (n_samples, n_levels).
    level_distances : NDArray[np.float64]
        The array of distances converted from the input array.
    """

    y_ndim = 2

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, level_distances: NDArray[np.float64], normalize_labels: bool = False) -> None:
        if level_distances.size == 0:
            raise ValueError("level_distances cannot be empty.")
        if np.any(level_distances < 0):
            raise ValueError("All level_distances must be non-negative.")
        self.level_distances = level_distances
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate that input has 2 dimensions and matches the configured levels.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of hierarchical labels.

        Returns
        -------
        NDArray[np.float64]
            The validated and potentially cast input array.

        Raises
        ------
        ValueError
            If `y` is empty, not 2D, or the number of columns (levels) does not
            match the length of `level_distances`.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 2:
            raise ValueError(
                f"Input 'y' must be 2-dimensional (n_samples, n_levels), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )

        expected_cols = len(self.level_distances)
        if y_proc.shape[1] != expected_cols:
            raise ValueError(
                f"Input 'y' must have {expected_cols} columns (matching level_distances length), "
                f"but got {y_proc.shape[1]} columns."
            )

        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the pairwise distance matrix based on the first diverging level.

        Identifies the index of the first column where $y_i$ and $y_j$ differ.
        The distance is then set to the value in `level_distances` at that index.
        If the rows are identical, the distance is 0.0.

        Parameters
        ----------
        y : NDArray[np.float64]
            A 2D array of shape (n_samples, n_levels).

        Returns
        -------
        NDArray[np.float64]
            A pairwise distance matrix encoding the hierarchical separation.
        """
        differences = y[:, None, :] != y[None, :, :]

        # np.argmax returns the first index of True (the first difference)
        first_diff_level = np.argmax(differences, axis=2)
        has_difference = np.any(differences, axis=2)

        # Map indices to distances; identical rows (no difference) get 0.0
        distance_matrix = np.where(has_difference, self.level_distances[first_diff_level], 0.0)

        return distance_matrix
