import numpy as np
import pytest

from smds import SupervisedMDS


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(stage_1="computed", manifold="log_linear", alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    return np.random.rand(100) * 10 + 0.1


def test_log_linear_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 1)
    assert not np.isnan(X_proj).any()
