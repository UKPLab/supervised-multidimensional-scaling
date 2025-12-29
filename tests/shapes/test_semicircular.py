import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.continuous_shapes.semicircular import SemicircularShape


@pytest.fixture
def engine() -> SupervisedMDS:
    # Semicircle is 2D
    return SupervisedMDS(n_components=2, manifold=SemicircularShape(), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    # Normalized to [0, 1]
    return np.random.rand(100)


def test_semicircular_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 2)
