import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from abc import ABC, abstractmethod
from typing import Self, Any


class BaseShape(BaseModel, BaseEstimator, TransformerMixin, ABC):  # type: ignore[misc]
    """
    Abstract base class for all manifold strategies.
    """
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def fit(self, y: NDArray[Any], y_ignored: None = None) -> Self:
        """
        'Fits' the strategy. Validates the input format.
        'y' here is the "X" data for this transformer.
        """
        self._validate_input(y)
        return self

    def transform(self, y: NDArray[Any]) -> NDArray[Any]:
        """
        Transforms labels y into an ideal distance matrix D.
        This is the main "Template Method".
        """
        y_proc = self._validate_input(y)
        
        D = self._compute_distances(y_proc)
        
        np.fill_diagonal(D, 0)
        return D

    def _validate_input(self, y: NDArray[Any]) -> NDArray[Any]:
        """
        Common validation for all strategies.
        Converts to array and checks if empty.
        Returns the processed array.
        """
        y_proc = np.asarray(y)
        
        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")
        return y_proc

    @abstractmethod
    def _compute_distances(self, y: NDArray[Any]) -> NDArray[Any]:
        """
        The specific distance computation logic to be implemented 
        by all concrete (sub)strategies.
        
        Subclasses MUST implement this method.
        They are responsible for validating input shape (e.g., 1D vs 2D).
        """
        raise NotImplementedError() 