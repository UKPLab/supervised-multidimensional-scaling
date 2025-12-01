import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes.log_linear import LogLinearShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(n_components=1, manifold=LogLinearShape(), alpha=0.1)


@pytest.fixture
def X() -> np.ndarray:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.ndarray:
    return np.random.rand(100) * 10 + 0.1


def test_log_linear_smoke(engine: SupervisedMDS, X: np.ndarray, y: np.ndarray) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 1)
    assert not np.isnan(X_proj).any()