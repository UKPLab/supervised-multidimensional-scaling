import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from abc import ABC, abstractmethod
from typing import Self


class BaseShape(BaseEstimator, TransformerMixin, ABC):  # type: ignore[misc]
    """
    Abstract base class for all manifold strategies.
    """

    def fit(self, y: NDArray[np.float64]) -> Self:
        """
        'Fits' the strategy. Validates the input format.
        'y' here is the "X" data for this transformer.
        """
        self._validate_input(y)
        return self

    def __call__(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Make BaseShape instances callable.
        Delegates to transform() 
        """
        #TODO: Do we really need?
        return self.transform(y)
    
    def transform(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transforms labels y into an ideal distance matrix D.
        This is the main "Template Method".
        """
        y_proc: NDArray[np.float64] = self._validate_input(y)
        n: int = len(y_proc)
        
        distance: NDArray[np.float64] = self._compute_distances(y_proc)
        
        if distance.shape != (n, n):
            raise ValueError(
                f"_compute_distances must return a square matrix of shape ({n}, {n}), "
                f"but got shape {distance.shape}."
            )
        
        np.fill_diagonal(distance, 0)
        return distance

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Common validation for all strategies.
        Converts to array, checks if empty, and validates shape.
        Returns the processed array.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.floating)
        
        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")
        
        if y_proc.ndim != 1:
            raise ValueError(
                f"Input 'y' must be 1-dimensional (n_samples,), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )
        
        return y_proc
    
    def _normalize_y(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize y to [0, 1] range with edge case handling.
        
        If all values are identical (max_y == min_y), returns zeros array.
        This ensures that subsequent distance calculations result in zero distances.
        
        Returns:
            y_norm: Normalized array in [0, 1] range, or zeros if all values are identical
        """
        max_y: np.float64 = np.max(y)
        min_y: np.float64 = np.min(y)
        
        if max_y == min_y:
            return np.zeros_like(y, dtype=float)
        
        y_norm: NDArray[np.float64] = (y - min_y) / (max_y - min_y)
        return y_norm

    @abstractmethod
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The specific distance computation logic to be implemented 
        by all concrete (sub)strategies.
        
        Subclasses MUST implement this method.
        They are responsible for validating input shape (e.g., 1D vs 2D).
        """
        raise NotImplementedError() 