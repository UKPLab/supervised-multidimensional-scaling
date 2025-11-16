from abc import ABC, abstractmethod

from numpy.typing import NDArray


class BaseStress(ABC):
    @abstractmethod
    def compute(self, X_high: NDArray, X_low: NDArray) -> float:
        """
        Compute the stress metric between high and low dimensional embeddings.

        Args:
            X_high: High-dimensional data points (n_samples, n_features_high)
            X_low: Low-dimensional embedding (n_samples, n_features_low)

        Returns:
            float: Computed stress value
        """
        raise NotImplementedError()
