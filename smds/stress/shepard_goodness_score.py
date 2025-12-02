from scipy.stats import spearmanr # type: ignore[import-untyped]
from numpy.typing import NDArray
import numpy as np


def shepard_goodness_score(D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
    """
    Compute the Shepard Goodness Score (Spearman's Rho of distance matrices).

    This metric evaluates the preservation of the global structure of the data
    by correlating the pairwise distances in the high-dimensional space with
    the pairwise distances in the low-dimensional space.

    Reference:
        Smelser et al., "Normalized Stress is Not Normalized: How to
        Interpret Stress Correctly", https://arxiv.org/html/2408.07724v1
    """
    rho, _ = spearmanr(D_high, D_low)

    return float(rho)