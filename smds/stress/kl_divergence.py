import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_consistent_length  # type: ignore[import-untyped]
from sklearn.utils._param_validation import validate_params  # type: ignore[import-untyped]

def _distances_to_probabilities(D: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """
    Helper to convert distance matrix to joint probability matrix using a Gaussian kernel.
    """
    D_sq = D ** 2
    np.fill_diagonal(D_sq, np.inf)
    
    # Gaussian Kernel
    P = np.exp(-D_sq / (2 * sigma**2))
    
    sum_P = np.sum(P, axis=1, keepdims=True)
    sum_P = np.maximum(sum_P, 1e-12) 

    P = P / sum_P
    
    P = (P + P.T) / (2 * P.shape[0])

    result: NDArray[np.float64] = np.maximum(P, 1e-12)
    return result

@validate_params(  # type: ignore[misc]
    {
        "d_true": ["array-like"],
        "d_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def kl_divergence_stress(
    d_true: NDArray[np.float64],
    d_pred: NDArray[np.float64],
    sigma: float = 1.0
) -> float:
    """
    Compute the Kullback-Leibler (KL) Divergence using Gaussian kernels for both P and Q.

    This metric measures the information loss when approximating the high-dimensional
    structure (P) with the low-dimensional structure (Q). Unlike standard t-SNE which
    uses a Student-t distribution for Q, this implementation uses a Gaussian kernel
    for both, corresponding to the metric denoted as :math:`KL_G` in Smelser et al. (2025).

    Parameters
    ----------
    d_true : array-like of shape (n_samples, n_samples)
        The target distance matrix (Ground Truth/Ideal).
        Must be a square, symmetric matrix of pairwise distances.

    d_pred : array-like of shape (n_samples, n_samples)
        The recovered distance matrix (Embedding).
        Must be a square, symmetric matrix of pairwise distances.

    sigma : float, default=1.0
        The standard deviation (width) of the Gaussian kernel.

    Returns
    -------
    kl_div : float
        The Kullback-Leibler divergence :math:`KL(P||Q)`.
        
    Notes
    -----
    The divergence is calculated as:

    .. math::
        KL(P||Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}

    Where both :math:`p_{ij}` and :math:`q_{ij}` are derived using Gaussian kernels:
    
    .. math::
        q_{ij}^G(\\alpha) = \\frac{\\exp(-\\alpha^2 ||y_i - y_j||^2)}{\\sum_{k \\neq l} \\exp(-\\alpha^2 ||y_k - y_l||^2)}
    
    (Assuming alpha=1 and incorporated into sigma).
    
    References
    ----------
    Smelser, K., Gunaratne, K., Miller, J., & Kobourov, S. (2025).
    "How Scale Breaks 'Normalized Stress' and KL Divergence: Rethinking Quality Metrics".
    arXiv preprint arXiv:2510.08660.
    """
    d_true = check_array(d_true, ensure_2d=True, dtype=np.float64)
    d_pred = check_array(d_pred, ensure_2d=True, dtype=np.float64)
    check_consistent_length(d_true, d_pred)

    if d_true.shape[0] != d_true.shape[1]:
        raise ValueError("d_true must be a square distance matrix.")
    if d_pred.shape[0] != d_pred.shape[1]:
        raise ValueError("d_pred must be a square distance matrix.")

    P = _distances_to_probabilities(d_true, sigma)
    Q = _distances_to_probabilities(d_pred, sigma)

    result: float = float(np.sum(P * np.log(P / Q)))
    return result