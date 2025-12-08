import numpy as np
from numpy.typing import NDArray

from smds.stress.base_stress import BaseStress


class NormalizedStress(BaseStress):
    """
    Computes the Supervised Normalized Stress.

    Reference: Equation 4 in "Shape Happens" paper.
    Formula: S := sum( (||Wx_i - Wx_j|| - d_hat_ij)^2 ) / sum( d_hat_ij^2 )
    """

    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
        """
        Computes stress between the ideal geometry (D_high) and recovered geometry (D_low).
        Handles incomplete matrices by masking out negative values in D_high.

        Args:
            D_high: The IDEAL distance matrix (d_hat_ij). Values < 0 are treated as undefined.
            D_low: The RECOVERED distance matrix (Euclidean dist in projection).
        """
        mask = D_high >= 0

        numerator: float = np.sum((D_low[mask] - D_high[mask]) ** 2)
        
        denominator: float = np.sum(D_high[mask] ** 2)

        if denominator == 0:
            return np.inf

        stress: float = numerator / denominator
        return stress