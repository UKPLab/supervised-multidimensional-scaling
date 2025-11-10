import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes.circular import CircularShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=2, manifold=CircularShape(radious=1.0), alpha=0.1)


@pytest.fixture
def X() -> np.ndarray:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.ndarray:
    return np.random.randint(0, 5, size=100)


def test_circular(engine: SupervisedMDS, X: np.ndarray, y: np.ndarray) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 2)
