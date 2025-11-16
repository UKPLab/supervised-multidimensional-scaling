import numpy as np
from scipy.spatial.distance import pdist

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

    def compute(self, X_high, X_low):
        # Compute pairwise distances (only upper triangle for efficiency)
        D_high = pdist(X_high, metric="euclidean")
        D_low = pdist(X_low, metric="euclidean")

        alpha = np.sum(D_high * D_low) / np.sum(D_high**2)

        residuals = D_high - alpha * D_low
        stress = np.sqrt(np.sum(residuals**2) / np.sum(D_high**2))
        return stress
