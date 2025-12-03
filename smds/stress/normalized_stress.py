import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import validate_params  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_consistent_length  # type: ignore[import-untyped]


@validate_params(  # type: ignore[misc]
    {
        "d_true": ["array-like"],
        "d_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def normalized_stress(d_true: NDArray[np.float64], d_pred: NDArray[np.float64]) -> float:
    """
    Compute the Normalized Stress between ideal and recovered geometries.

    This metric quantifies the preservation of pairwise distances by comparing
    the squared differences relative to the magnitude of the ideal distances.

    Parameters
    ----------
    d_true : array-like of shape (n_pairs,)
        The ideal/target distance matrix (D_high).
        Can be a flattened array of pairwise distances.

    d_pred : array-like of shape (n_pairs,)
        The recovered/embedding distance matrix (D_low).
        Can be a flattened array of pairwise distances.

    Returns
    -------
    stress : float
        The calculated normalized stress value.

    Notes
    -----
    The formula implemented is:

    .. math::

        S = \\frac{\\sum (d_{pred} - d_{true})^2}{\\sum d_{true}^2}

    Note that typically 'Normalized Stress' includes a square root operation.
    This implementation returns the squared variant (variance explained ratio).

    References
    ----------
    - Smelser, K., Miller, J., & Kobourov, S. (2024). "Normalized Stress is Not
    Normalized: How to Interpret Stress Correctly". arXiv preprint arXiv:2408.07724.
    """
    d_true = check_array(d_true, ensure_2d=False, dtype=np.float64)
    d_pred = check_array(d_pred, ensure_2d=False, dtype=np.float64)
    check_consistent_length(d_true, d_pred)

    numerator = np.sum((d_pred - d_true) ** 2)
    denominator = np.sum(d_true ** 2)

    if denominator == 0:
        return np.inf
    result: float = float(numerator / denominator)

    return result
