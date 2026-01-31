import numpy as np
import pytest

from smds import ComputedSMDSParametrization, SupervisedMDS
from smds.shapes.continuous_shapes import KleinBottleShape


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(ComputedSMDSParametrization(n_components=2, manifold=KleinBottleShape()), alpha=0.1)


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


def test_klein_bottle_rejects_high_dimensions(engine: SupervisedMDS, X: np.typing.NDArray[np.float64]) -> None:
    y_high_dim = np.random.rand(100, 5)
    with pytest.raises(ValueError, match="Klein Bottle requires exactly 2 dimensions"):
        engine.fit(X, y_high_dim)
