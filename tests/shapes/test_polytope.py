import numpy as np
import pytest

from smds import SupervisedMDS
from smds.shapes.discrete_shapes import PolytopeShape


@pytest.fixture
def engine() -> SupervisedMDS:
    # We use PolytopeShape with 10 neighbors
    return SupervisedMDS(n_components=2, manifold=PolytopeShape(n_neighbors=10), alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 10)


@pytest.fixture
def y_swiss_roll() -> np.typing.NDArray[np.float64]:
    """
    Generates a Swiss Roll dataset (classic manifold learning benchmark).
    """
    n_samples = 100
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    h = 21 * np.random.rand(1, n_samples)

    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)

    # Shape: (100, 3)
    return np.concatenate((x, y, z)).T


def test_polytope_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y_swiss_roll: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y_swiss_roll)
    X_proj = engine.transform(X)

    assert X_proj.shape == (100, 2)
