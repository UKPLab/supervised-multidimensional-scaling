from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]


class BaseShape(BaseEstimator, ABC):  # type: ignore[misc]
    """
    General abstraction for shapes (manifolds).
    """

    @property
    @abstractmethod
    def normalize_labels(self) -> bool:
        """
        Flag to normalize labels.
        Should default to True unless specified by the shape.
        """
        pass

    @staticmethod
    def _do_normalize_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Default implementation - 0-1 normalization.
        Override for your Shape if needed.
        """
        max_y = np.max(y)
        min_y = np.min(y)
        if max_y == min_y:
            return np.zeros_like(y, dtype=float)
        return (y - min_y) / (max_y - min_y)

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Make BaseShape instances callable.
        Transforms labels y into an ideal distance matrix D.
        This is the main "Template Method".
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
        Common validation for all strategies.
        Converts to array, checks if empty, and validates shape.
        Returns the processed array.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 1:
            raise ValueError(
                f"Input 'y' must be 1-dimensional (n_samples,), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )

        return y_proc

    @abstractmethod
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The specific distance computation logic to be implemented
        by all concrete (sub)strategies..
        """
        raise NotImplementedError()
