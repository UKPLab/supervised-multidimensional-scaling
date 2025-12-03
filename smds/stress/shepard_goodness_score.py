import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils._param_validation import validate_params

@validate_params(
    {
        "d_true": ["array-like"],
        "d_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def shepard_goodness_stress(d_true: NDArray[np.float64], d_pred: NDArray[np.float64]) -> float:
    """
    Compute the Shepard Goodness Score (Spearman's Rho) on pairwise distances.

    Parameters
    ----------
    d_true : array-like of shape (n_pairs,)
        The target dissimilarities (D_high/D_ideal).

    d_pred : array-like of shape (n_pairs,)
        The embedding distances (D_low/D_pred).

    Returns
    -------
    score : float
        Spearman's rank correlation coefficient (rho).

    References
    ----------
    - Smelser, K., Miller, J., & Kobourov, S. (2024). "Normalized Stress is Not
    Normalized: How to Interpret Stress Correctly". arXiv preprint arXiv:2408.07724.
    """
    d_true = check_array(d_true, ensure_2d=False, dtype=np.float64)
    d_pred = check_array(d_pred, ensure_2d=False, dtype=np.float64)
    check_consistent_length(d_true, d_pred)

    correlation = spearmanr(d_true, d_pred)

    result: float = float(correlation[0])
    return result
