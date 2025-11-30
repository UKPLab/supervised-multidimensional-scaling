import numpy as np
from numpy.typing import NDArray

from smds.stress.base_stress import BaseStress


class KLDivergence(BaseStress):
    """
    Computes Kullback-Leibler (KL) Divergence between the Ideal distribution (P)
    and the Recovered distribution (Q).
    """

    def _distances_to_probabilities(self, D: NDArray[np.float64], sigma: float = 1.0) -> NDArray[np.float64]:
        """
        Convert a distance matrix to a probability matrix using a Gaussian kernel.
        """
        D_sq = D ** 2       
        np.fill_diagonal(D_sq, np.inf)
        P = np.exp(-D_sq / (2 * sigma**2))
        
        sum_P = np.sum(P, axis=1, keepdims=True)
        sum_P = np.maximum(sum_P, 1e-12)

        P = P / sum_P
        P = (P + P.T) / (2 * P.shape[0])

        return np.maximum(P, 1e-12)

    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
        """
        Args:
            D_high: Ideal distances (Ground Truth)
            D_low: Recovered distances
        """
        # Convert distances to probability distributions P (Ideal) and Q (Recovered)
        P = self._distances_to_probabilities(D_high, sigma=1.0)
        Q = self._distances_to_probabilities(D_low, sigma=1.0)

        kl_div: float = np.sum(P * np.log(P / Q))

        return kl_div