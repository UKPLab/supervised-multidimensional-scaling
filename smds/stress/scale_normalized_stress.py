import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils._param_validation import validate_params

@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def scale_normalized_stress(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Compute the scale-normalized stress between true dissimilarities and embedding distances.

    This metric calculates stress by first finding an optimal scaling factor
    alpha to match the scale of the embedding distances to the true dissimilarities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_samples)
        The original high-dimensional dissimilarities (D_high).
        These act as the target structure.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_samples)
        The low-dimensional embedding distances (D_low).
        These are the values being evaluated.

    Returns
    -------
    stress : float
        The calculated scale-normalized stress value. Returns np.inf if the
        norm of the prediction is zero.

    Notes
    -----
    The calculation involves finding a scaling factor :math:`\\alpha` such that:

    .. math::

        \\alpha = \\frac{\\sum (y_{true} \\cdot y_{pred})}{\\sum y_{pred}^2}

    The stress is then computed as:

    .. math::

        \\sigma = \\sqrt{\\frac{\\sum (y_{true} - \\alpha \\cdot y_{pred})^2}{\\sum y_{true}^2}}

    References
    ----------
    Smelser et al., "Normalized Stress is Not Normalized: How to
    Interpret Stress Correctly", https://arxiv.org/html/2408.07724v1
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=np.float64)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=np.float64)
    check_consistent_length(y_true, y_pred)

    denominator_alpha = np.sum(y_pred ** 2)

    if denominator_alpha == 0:
        return np.inf

    alpha = np.sum(y_true * y_pred) / denominator_alpha

    residuals = y_true - (alpha * y_pred)
    
    return np.sqrt(np.sum(residuals ** 2) / np.sum(y_true ** 2))