import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes import CircularShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=CircularShape(radious=1.0), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.int64]:
    return np.random.randint(0, 5, size=100)


def test_circular(engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.int64]) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 2)
