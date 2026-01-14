from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]


class BaseShape(BaseEstimator, ABC):  # type: ignore[misc]
    """
    Abstract base class for defining manifold shapes.

    This class serves as a template for transforming input labels (y) into
    a pairwise distance matrix that represents the geometry of a specific
    shape (manifold). It handles input validation, optional normalization,
    and structural integrity checks on the output matrix.

    Subclasses must implement:
    - `y_ndim`: Property defining expected input dimensionality.
    - `normalize_labels`: Property flag for normalization behavior.
    - `_compute_distances`: The core logic for mapping labels to distances.

    Attributes
    ----------
    y_ndim : int
        Abstract property. The expected dimensionality of the input labels.
    normalize_labels : bool
        Abstract property. Whether to normalize inputs before computation.
    """

    @property
    @abstractmethod
    def y_ndim(self) -> int:
        """
        Get the required dimensionality of the input labels `y`.

        Returns
        -------
        int
            The number of dimensions expected for the input array.
            - 1: 1D array (e.g., time series, clusters).
            - 2: 2D array (e.g., lat/lon coordinates, hierarchical levels).
        """
        pass

    @property
    @abstractmethod
    def normalize_labels(self) -> bool:
        """
        Get the flag indicating whether input labels should be normalized.

        Returns
        -------
        bool
            True if `_do_normalize_labels` should be called in `__call__`,
            False otherwise.
        """
        pass

    @staticmethod
    def _do_normalize_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform default Min-Max normalization on the input labels.

        Maps the input values to the range [0, 1]. If the input has no
        variance (all values are equal), a zero array is returned.

        Parameters
        ----------
        y : NDArray[np.float64]
            The input label array.

        Returns
        -------
        NDArray[np.float64]
            The normalized array.
        """
        max_y = np.max(y)
        min_y = np.min(y)
        if max_y == min_y:
            return np.zeros_like(y, dtype=float)
        return (y - min_y) / (max_y - min_y)

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the pairwise distance matrix for the given labels.

        This is the main entry point (Template Method). It performs the
        following steps:
        1. Validates the input `y` (dimensions and emptiness).
        2. Normalizes `y` if `self.normalize_labels` is True.
        3. Calls the subclass implementation of `_compute_distances`.
        4. Validates that the output is a square matrix.
        5. Enforces a zero diagonal.

        Parameters
        ----------
        y : NDArray[np.float64]
            The input labels or coordinates used to position points on the
            manifold. Must match `self.y_ndim`.

        Returns
        -------
        NDArray[np.float64]
            A square matrix of shape (n_samples, n_samples) containing
            pairwise Euclidean distances on the defined manifold.

        Raises
        ------
        ValueError
            If `_compute_distances` returns a matrix with incorrect dimensions.
        """
        y_proc: NDArray[np.float64] = self._validate_input(y)
        n: int = len(y_proc)

        if self.normalize_labels:
            y_proc = self._do_normalize_labels(y_proc)

        distance: NDArray[np.float64] = self._compute_distances(y_proc)

        if distance.shape != (n, n):
            raise ValueError(
                f"_compute_distances must return a square matrix of shape ({n}, {n}), but got shape {distance.shape}."
            )

        np.fill_diagonal(distance, 0)
        return distance

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input array dimensions and content.

        Parameters
        ----------
        y : NDArray[np.float64]
            The raw input array.

        Returns
        -------
        NDArray[np.float64]
            The validated array, cast to float64.

        Raises
        ------
        ValueError
            If the input array is empty or does not match `self.y_ndim`.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != self.y_ndim:
            raise ValueError(
                f"Input 'y' must be {self.y_ndim}-dimensional (n_samples,), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )

        return y_proc

    @abstractmethod
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the specific pairwise distances for the implemented shape.

        This abstract method must be overridden by subclasses to define the
        geometry of the manifold.

        Parameters
        ----------
        y : NDArray[np.float64]
            The processed (and potentially normalized) input labels.

        Returns
        -------
        NDArray[np.float64]
            A square pairwise distance matrix.
        """
        raise NotImplementedError()
