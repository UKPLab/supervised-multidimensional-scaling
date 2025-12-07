import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.utils._param_validation import validate_params  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_consistent_length  # type: ignore[import-untyped]


@validate_params(  # type: ignore[misc]
    {
        "d_true": ["array-like"],
        "d_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def non_metric_stress(d_true: NDArray[np.float64], d_pred: NDArray[np.float64]) -> float:
    """
    Compute the non-metric stress between true dissimilarities and embedding distances.

    Parameters
    ----------
    d_true : array-like of shape (n_pairs,)
        The target dissimilarities (D_high/D_ideal). Expected to be a 1D array
        of flattened pairwise distances.

    d_pred : array-like of shape (n_pairs,)
        The embedding distances (D_low/D_pred). Expected to be a 1D array
        of flattened pairwise distances.

    Returns
    -------
    stress : float
        The non-metric stress value.
    References
    ----------
    - Smelser, K., Miller, J., & Kobourov, S. (2024). "Normalized Stress is Not
    Normalized: How to Interpret Stress Correctly". arXiv preprint arXiv:2408.07724.
    """
    d_true = check_array(d_true, ensure_2d=False, dtype=np.float64)
    d_pred = check_array(d_pred, ensure_2d=False, dtype=np.float64)
    check_consistent_length(d_true, d_pred)

    ir = IsotonicRegression(increasing=True)
    d_hat = ir.fit_transform(d_true, d_pred)

    numerator = np.sum((d_hat - d_pred) ** 2)
    denominator = np.sum(d_pred**2)

    if denominator == 0:
        return np.inf

    result: float = float(numerator / denominator)
    return result
