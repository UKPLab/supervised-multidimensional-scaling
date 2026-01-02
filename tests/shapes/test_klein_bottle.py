import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes import KleinBottleShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=KleinBottleShape(), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    return np.random.rand(100, 2)


def test_klein_bottle_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 2)
