from smds.stress.base_stress import BaseStress
from numpy.typing import NDArray
from zadu.measures import spearman_rho
import numpy as np
from scipy.spatial.distance import pdist


class ShepardGoodnessScore(BaseStress):
    def compute(self, X_high: NDArray[np.float64], X_low: NDArray[np.float64]) -> float:
        dist_high = pdist(X_high)
        dist_low = pdist(X_low)

        res = spearman_rho.measure(X_high, X_low, (dist_high, dist_low))
        return res['spearman_rho']