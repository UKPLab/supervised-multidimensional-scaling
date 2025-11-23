import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from sklearn.isotonic import IsotonicRegression

from smds.stress.base_stress import BaseStress


class NonMetricStress(BaseStress):
    def compute(self, X_high: NDArray, X_low: NDArray) -> float:
        D_high: NDArray[np.float64] = pdist(X_high, metric="euclidean")
        D_low: NDArray[np.float64] = pdist(X_low, metric="euclidean")

        ir: IsotonicRegression = IsotonicRegression(increasing=True)
        d_hat: NDArray[np.float64] = ir.fit_transform(D_high, D_low)

        numerator: float = np.sum((d_hat - D_low) ** 2)

        denominator: float = np.sum(D_low**2)

        if denominator == 0:
            return np.inf

        stress: float = numerator / denominator
        return stress
