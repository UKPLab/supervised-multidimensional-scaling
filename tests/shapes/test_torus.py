import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes import TorusShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=TorusShape(), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    return np.random.rand(100, 2)


def test_torus_smoke(engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 2)
