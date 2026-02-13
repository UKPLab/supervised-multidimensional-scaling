import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.estimator_checks import check_estimator  # type: ignore[import-untyped]

from smds.hsmds import HybridSMDS


def dummy_manifold_func(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Dummy manifold function that works with ANY y (for tests).
    Simply returns euclidean distance on y, regardless of what y is.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    result: NDArray[np.float64] = squareform(pdist(y))
    return result


class DummyReducer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    A minimal reducer that is scikit-learn compliant.
    Serves as a placeholder for e.g. PLSRegression.
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "DummyReducer":
        self.n_features_in_ = X.shape[1]
        self.output_dim_ = y.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not hasattr(self, "_is_fitted"):
            raise RuntimeError("Not fitted")
        return np.zeros((X.shape[0], self.output_dim_))

    def inverse_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros((X.shape[0], self.n_features_in_))


def test_sklearn_compatibility() -> None:
    """
    Tests HybridSMDS with sklearn's check_estimator.
    This ensures that HybridSMDS meets all sklearn compatibility requirements.
    """
    estimator = HybridSMDS(manifold=dummy_manifold_func, reducer=DummyReducer(), n_components=2)

    try:
        check_estimator(estimator)
    except (AssertionError, ValueError, TypeError, AttributeError) as e:
        pytest.fail(f"Error in check_estimator: {e}")
