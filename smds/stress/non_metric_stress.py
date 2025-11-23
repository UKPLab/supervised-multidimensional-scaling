import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from smds.stress.base_stress import BaseStress


class NonMetricStress(BaseStress):
    """
    Notes:
    - Time complexity: ... # TODO: Decide if we want to include such things. IMO could be interesting
    - Space complexity: ...

    Reference:
        Smelser et al., "Normalized Stress is Not Normalized: How to
        Interpret Stress Correctly", https://arxiv.org/html/2408.07724v1
    """
    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:

        ir: IsotonicRegression = IsotonicRegression(increasing=True)
        d_hat: NDArray[np.float64] = ir.fit_transform(D_high, D_low)

        numerator: float = np.sum((d_hat - D_low) ** 2)

        denominator: float = np.sum(D_low**2)

        if denominator == 0:
            return np.inf

        stress: float = numerator / denominator
        return stress
