import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]
from sklearn.utils.estimator_checks import check_estimator  # type: ignore[import-untyped]

from smds.smds import ComputedStage1, SupervisedMDS

# todo: add same for Stage1SMDSTransformer


def dummy_manifold_func(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Dummy manifold function that works with ANY y (for tests).
    Simply returns euclidean distance on y, regardless of what y is.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    result: NDArray[np.float64] = squareform(pdist(y))
    return result


def test_sklearn_compatibility() -> None:
    """
    Tests SupervisedMDS with sklearn's check_estimator.
    This ensures that SupervisedMDS meets all sklearn compatibility requirements.
    """
    # fixme: dummy_manifold_func has to be an instance of BaseShape
    estimator = SupervisedMDS(ComputedStage1(manifold=dummy_manifold_func, n_components=2))

    try:
        check_estimator(estimator)
    except (AssertionError, ValueError, TypeError, AttributeError) as e:
        pytest.fail(f"Error in check_estimator: {e}")
