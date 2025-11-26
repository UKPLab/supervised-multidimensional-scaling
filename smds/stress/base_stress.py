from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseStress(ABC):
    @abstractmethod
    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
        """
        Compute the stress metric between high and low dimensional embeddings.

        Args:
            D_high: Pairwise distances in high-dimensional space (1D array of shape (n_pairs,))
            D_low: Pairwise distances in low-dimensional space (1D array of shape (n_pairs,))

        Returns:
            float: Computed stress value
        """
        raise NotImplementedError()
