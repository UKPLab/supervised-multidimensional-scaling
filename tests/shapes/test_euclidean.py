import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

from smds import SupervisedMDS
from smds.shapes.continuous_shapes import EuclideanShape

N_LATENT = 5


@pytest.fixture
def engine() -> SupervisedMDS:
    return SupervisedMDS(parametrization="computed", manifold="euclidean", alpha=0.1)


@pytest.fixture
def X() -> np.typing.NDArray[np.float64]:
    return np.random.randn(100, 20)


@pytest.fixture
def y() -> np.typing.NDArray[np.float64]:
    return np.random.rand(100) * 10


@pytest.fixture
def euclidean_data_5d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    20D observations carrying a latent 5D Euclidean point cloud.

    ``y`` holds the latent coordinates directly, so a well-fitted 5-component
    embedding should reproduce the latent pairwise distances up to scale.
    """
    rng = np.random.default_rng(42)
    y_latent = rng.standard_normal((100, N_LATENT))

    projection = rng.standard_normal((N_LATENT, 20))
    X_high = y_latent @ projection + rng.normal(scale=0.1, size=(100, 20))

    return X_high, y_latent


def test_euclidean_smoke(
    engine: SupervisedMDS, X: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64]
) -> None:
    engine.fit(X, y)
    X_proj = engine.transform(X)
    assert X_proj.shape == (100, 1)


# =============================================================================
# N-DIMENSIONAL LABELS
# =============================================================================


def test_euclidean_accepts_scalar_and_coordinate_labels() -> None:
    """``y_ndim`` declares both (n_samples,) and (n_samples, n_features)."""
    assert EuclideanShape.y_ndim == (1, 2)


@pytest.mark.parametrize("n_features", [1, 3, 7])
def test_shape_matches_cdist(n_features: int) -> None:
    """The distance matrix is the true Euclidean metric in n dimensions."""
    y_nd = np.random.default_rng(0).random((15, n_features))
    shape = EuclideanShape(normalize_labels=False)

    assert np.allclose(shape(y_nd), cdist(y_nd, y_nd, metric="euclidean"))


def test_shape_treats_column_vector_as_scalar_labels() -> None:
    """(n_samples,) and (n_samples, 1) describe the same points on a line."""
    y_1d = np.random.default_rng(0).random(15)
    shape = EuclideanShape(normalize_labels=False)

    assert np.allclose(shape(y_1d), shape(y_1d.reshape(-1, 1)))


def test_shape_rejects_three_dimensional_labels() -> None:
    shape = EuclideanShape()

    with pytest.raises(ValueError, match="must be 1- or 2-dimensional"):
        shape(np.zeros((4, 2, 2)))


def test_smds_rejects_three_dimensional_labels(X: NDArray[np.float64]) -> None:
    """The estimator rejects unusable label ranks before reaching the shape."""
    engine = SupervisedMDS(manifold="euclidean", alpha=0.1)

    with pytest.raises(ValueError, match="must be 1- or 2-dimensional"):
        engine.fit(X, np.zeros((100, 2, 2)))


def test_smds_fits_coordinate_labels(euclidean_data_5d: tuple[NDArray[np.float64], NDArray[np.float64]]) -> None:
    """The motivating case: 5D labels with a matching 5D embedding."""
    X_high, y_latent = euclidean_data_5d
    engine = SupervisedMDS(manifold="euclidean", n_components=N_LATENT, alpha=0.1).fit(X_high, y_latent)

    assert engine.Y_.shape == (100, N_LATENT)
    assert engine.W_.shape == (N_LATENT, 20)
    assert engine.transform(X_high).shape == (100, N_LATENT)


def test_embedding_reproduces_latent_distances(
    euclidean_data_5d: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    ``Y_`` is a rigid re-embedding of the latent cloud, so its pairwise
    distances match the latent ones (Euclidean labels are normalized by a
    single global min-max factor, hence the shared rescaling).
    """
    X_high, y_latent = euclidean_data_5d
    engine = SupervisedMDS(manifold="euclidean", n_components=N_LATENT, alpha=0.1).fit(X_high, y_latent)

    d_embedded = cdist(engine.Y_, engine.Y_)
    d_latent = cdist(y_latent, y_latent)
    scale = d_latent.max() / d_embedded.max()

    assert np.allclose(d_embedded * scale, d_latent)


def test_coordinate_labels_beat_the_scalar_default(
    euclidean_data_5d: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    ``_build_manifold`` fixes the euclidean default at 1 for scalar labels, so
    coordinate labels need an explicit ``n_components`` to avoid truncation.
    """
    X_high, y_latent = euclidean_data_5d

    default = SupervisedMDS(manifold="euclidean", alpha=0.1).fit(X_high, y_latent)
    widened = SupervisedMDS(manifold="euclidean", n_components=N_LATENT, alpha=0.1).fit(X_high, y_latent)

    assert default.Y_.shape == (100, 1)
    assert np.all(widened.Y_.std(axis=0) > 1e-6)
    assert widened.score(X_high, y_latent) > default.score(X_high, y_latent)
