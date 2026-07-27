"""
Tests for the ``n_components`` override on SupervisedMDS.

``n_components=None`` (the default) keeps each built-in manifold's natural
embedding dimension; an explicit value overrides it. See the manifold factory
table in ``SupervisedMDS._build_manifold``.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]

from smds import HybridSMDS, SupervisedMDS

N_SAMPLES = 80
N_FEATURES = 10


@pytest.fixture
def X() -> NDArray[np.float64]:
    return np.random.default_rng(0).standard_normal((N_SAMPLES, N_FEATURES))


@pytest.fixture
def y_scalar() -> NDArray[np.float64]:
    return np.random.default_rng(1).random(N_SAMPLES) * 2 * np.pi


@pytest.fixture
def y_2d() -> NDArray[np.float64]:
    return np.random.default_rng(2).random((N_SAMPLES, 2))


@pytest.fixture
def y_clusters() -> NDArray[np.float64]:
    """Six clusters: the distance matrix has rank 5, well above the default of 2."""
    return np.random.default_rng(3).integers(0, 6, N_SAMPLES).astype(float)


# =============================================================================
# DEFAULT (n_components=None) PRESERVES THE MANIFOLD'S NATURAL DIMENSION
# =============================================================================


@pytest.mark.parametrize(
    ("manifold", "expected", "y_fixture"),
    [
        ("euclidean", 1, "y_scalar"),
        ("log_linear", 1, "y_scalar"),
        ("circular", 2, "y_scalar"),
        ("cluster", 2, "y_clusters"),
        ("spherical", 3, "y_2d"),
        ("klein_bottle", 4, "y_2d"),
    ],
)
def test_default_uses_manifold_dimension(
    X: NDArray[np.float64],
    manifold: str,
    expected: int,
    y_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    y = request.getfixturevalue(y_fixture)
    engine = SupervisedMDS(manifold=manifold, alpha=0.1).fit(X, y)

    assert engine.n_components is None
    assert engine.Y_.shape == (N_SAMPLES, expected)
    assert engine.transform(X).shape == (N_SAMPLES, expected)


def test_default_is_none_and_unmodified_by_init() -> None:
    """Parameters must be stored untouched by __init__, per the sklearn API."""
    assert SupervisedMDS().n_components is None
    assert SupervisedMDS(n_components=5).n_components == 5


# =============================================================================
# EXPLICIT OVERRIDE
# =============================================================================


@pytest.mark.parametrize("k", [1, 3, 5])
def test_override_sets_embedding_dimension(X: NDArray[np.float64], y_clusters: NDArray[np.float64], k: int) -> None:
    engine = SupervisedMDS(manifold="cluster", n_components=k, alpha=0.1).fit(X, y_clusters)

    assert engine.Y_.shape == (N_SAMPLES, k)
    assert engine.W_.shape == (k, N_FEATURES)
    assert engine.transform(X).shape == (N_SAMPLES, k)


def test_override_retains_geometry_the_default_truncates(
    X: NDArray[np.float64], y_clusters: NDArray[np.float64]
) -> None:
    """
    The motivating case: a discrete shape whose distance matrix out-ranks the
    hard-coded default, so extra components carry real structure.
    """
    default = SupervisedMDS(manifold="cluster", alpha=0.1).fit(X, y_clusters)
    widened = SupervisedMDS(manifold="cluster", n_components=5, alpha=0.1).fit(X, y_clusters)

    # Every axis of the wider embedding is non-degenerate, not zero padding.
    assert np.all(widened.Y_.std(axis=0) > 1e-6)

    # The first components agree up to sign; the extra ones are new information.
    assert np.allclose(np.abs(widened.Y_[:, :2]), np.abs(default.Y_), atol=1e-8)

    # Retaining more of the distance matrix fits it better (higher score is better).
    assert widened.score(X, y_clusters) > default.score(X, y_clusters)


def test_override_beyond_rank_pads_with_zeros(X: NDArray[np.float64], y_scalar: NDArray[np.float64]) -> None:
    """
    Classical MDS cannot exceed the rank of the distance matrix. A circle spans
    two dimensions, so the trailing components are numerically zero (residual
    eigenvalues at the noise floor) rather than extra geometry.
    """
    engine = SupervisedMDS(manifold="circular", n_components=4, alpha=0.1).fit(X, y_scalar)

    assert engine.Y_.shape == (N_SAMPLES, 4)
    assert np.all(engine.Y_[:, :2].std(axis=0) > 1e-2)
    assert np.all(np.abs(engine.Y_[:, 2:]) < 1e-6)


# =============================================================================
# VALIDATION
# =============================================================================


@pytest.mark.parametrize("bad", [0, -3, 1.5, "2", True, np.array([2])])
def test_invalid_n_components_raises(X: NDArray[np.float64], y_scalar: NDArray[np.float64], bad: Any) -> None:
    engine = SupervisedMDS(manifold="circular", n_components=bad, alpha=0.1)

    with pytest.raises(ValueError, match="n_components must be a positive integer or None"):
        engine.fit(X, y_scalar)


def test_numpy_integer_is_accepted(X: NDArray[np.float64], y_clusters: NDArray[np.float64]) -> None:
    # Annotated as int, but numpy scalars leak in from things like y.shape[1] or
    # rng draws, so they are accepted at runtime.
    engine = SupervisedMDS(
        manifold="cluster",
        n_components=np.int64(3),  # type: ignore[arg-type]
        alpha=0.1,
    ).fit(X, y_clusters)

    assert engine.Y_.shape == (N_SAMPLES, 3)


# =============================================================================
# SKLEARN COMPATIBILITY
# =============================================================================


def test_get_params_and_clone_roundtrip() -> None:
    engine = SupervisedMDS(manifold="cluster", n_components=5)

    assert engine.get_params()["n_components"] == 5
    assert clone(engine).get_params() == engine.get_params()


def test_set_params(X: NDArray[np.float64], y_clusters: NDArray[np.float64]) -> None:
    engine = SupervisedMDS(manifold="cluster", alpha=0.1)
    engine.set_params(n_components=4).fit(X, y_clusters)

    assert engine.Y_.shape == (N_SAMPLES, 4)


# =============================================================================
# INTERACTION WITH OTHER PARAMETRIZATIONS AND SUBCLASSES
# =============================================================================


def test_user_provided_parametrization_ignores_n_components(
    X: NDArray[np.float64], y_scalar: NDArray[np.float64]
) -> None:
    """``user_provided`` infers its dimensionality from the coordinates."""
    with pytest.warns(UserWarning, match="manifold value is ignored"):
        engine = SupervisedMDS(parametrization="user_provided", n_components=7, alpha=0.1).fit(X, y_scalar)

    assert engine.Y_.shape == (N_SAMPLES, 1)


def test_hybrid_smds_keeps_lower_bound_semantics(X: NDArray[np.float64], y_2d: NDArray[np.float64]) -> None:
    """
    Regression guard: HybridSMDS.n_components is a floor that Y_ is zero-padded
    up to, not an override of the manifold's dimension, so a 3D spherical
    embedding survives n_components=2.
    """
    hybrid = HybridSMDS(manifold="spherical", n_components=2, reducer=PLSRegression(n_components=2))
    hybrid.fit(X, y_2d)

    assert hybrid.Y_.shape == (N_SAMPLES, 3)
