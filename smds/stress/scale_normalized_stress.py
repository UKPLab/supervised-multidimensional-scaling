import numpy as np
from numpy.typing import NDArray

from smds.stress.base_stress import BaseStress


class ScaleNormalizedStress(BaseStress):
    """
    Notes:
    - Time complexity: ... # TODO: Decide if we want to include such things. IMO could be interesting
    - Space complexity: ...

    Reference:
        Smelser et al., "Normalized Stress is Not Normalized: How to
        Interpret Stress Correctly", https://arxiv.org/html/2408.07724v1
    """

    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
        denominator_alpha: float = np.sum(D_low**2)

        if denominator_alpha == 0:
            return np.inf

        alpha: float = np.sum(D_high * D_low) / denominator_alpha

        residuals: NDArray[np.float64] = D_high - (alpha * D_low)

        stress: float = np.sqrt(np.sum(residuals**2) / np.sum(D_high**2))

        return stress
