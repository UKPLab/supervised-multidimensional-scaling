from abc import ABC, abstractmethod

from numpy.typing import NDArray


class BaseStress(ABC):
    @abstractmethod
    def compute(self, D_high: NDArray, D_low: NDArray) -> float:
        """
        Compute the stress metric between high and low dimensional embeddings.

        Args:
            D_high: Pairwise distances in high-dimensional space (1D array of shape (n_pairs,))
            D_low: Pairwise distances in low-dimensional space (1D array of shape (n_pairs,))

        Returns:
            float: Computed stress value
        """
        raise NotImplementedError()
