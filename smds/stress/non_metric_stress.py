import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils._param_validation import validate_params

@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def non_metric_stress(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """
    Compute the non-metric stress between true dissimilarities and embedding distances.

    This metric quantifies how well the ordinal relationship of the distances in
    the low-dimensional embedding preserves the ordinal relationship of the
    original high-dimensional dissimilarities using Isotonic Regression.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_samples)
        The original high-dimensional dissimilarities (D_high).
        These act as the target order for the isotonic regression.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_samples)
        The low-dimensional embedding distances (D_low).
        These are the values being transformed to fit the order of y_true.

    Returns
    -------
    stress : float
        The calculated non-metric stress value. Returns np.inf if the
        denominator (sum of squared distances) is zero.

    Notes
    -----
    The formula used corresponds to a normalized stress calculation:

    .. math::

        \sigma = \frac{\sum (d_{hat} - d_{low})^2}{\sum d_{low}^2}

    Where :math:`d_{hat}` are the values fitted by Isotonic Regression.

    References
    ----------
    Smelser et al., "Normalized Stress is Not Normalized: How to
    Interpret Stress Correctly", https://arxiv.org/html/2408.07724v1
    """
    y_true = check_array(y_true, ensure_2d=False, dtype=np.float64)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=np.float64)
    check_consistent_length(y_true, y_pred)

    ir = IsotonicRegression(increasing=True)
    d_hat = ir.fit_transform(y_true, y_pred)

    numerator = np.sum((d_hat - y_pred) ** 2)
    denominator = np.sum(y_pred ** 2)

    if denominator == 0:
        return np.inf

    return numerator / denominator