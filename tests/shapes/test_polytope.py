import numpy as np
import pytest

from smds import SupervisedMDS


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(manifold="polytope", alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(40, 10)


@pytest.fixture
def y_clusters() -> np.typing.NDArray[np.float64]:
    """
    Generates 4 distinct cluster categories (10 points per cluster).
    Polytope shape expects 1D categorical labels.
    """
    y = np.repeat([0, 1, 2, 3], 10).astype(np.float64)
    return y


def test_polytope_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y_clusters: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y_clusters)
    X_proj = engine.transform(X)

    assert X_proj.shape == (40, 3)
