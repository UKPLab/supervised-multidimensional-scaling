import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes.euclidean import EuclideanShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=1, manifold=EuclideanShape(), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    return np.random.rand(100) * 10


def test_euclidean_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 1)
