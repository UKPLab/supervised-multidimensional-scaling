import numpy as np
import pytest
from numpy.typing import NDArray

from smds import ComputedSMDSParametrization, SupervisedMDS
from smds.shapes.discrete_shapes import HierarchicalShape


@pytest.fixture
def random_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_samples = 50
    n_levels = 3
    X: NDArray[np.float64] = np.random.randn(n_samples, 10)
    y: NDArray[np.float64] = np.random.randint(0, 3, size=(n_samples, n_levels)).astype(float)
    return X, y


@pytest.fixture
def smds_engine() -> SupervisedMDS:
    return SupervisedMDS(
        ComputedSMDSParametrization(
            n_components=2, manifold=HierarchicalShape(level_distances=np.array([100.0, 10.0, 1.0]))
        )
    )


@pytest.mark.smoke
def test_hierarchical_smoke_test(
    random_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    A simple "smoke test" to ensure HierarchicalShape can be fit and transformed without errors
     and produces an output of the correct shape.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = random_data

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    n_samples = X.shape[0]
    n_components = smds_engine.stage_1.n_components

    assert X_proj.shape == (n_samples, n_components), (
        f"Output shape is incorrect. "
        f"Expected (n_samples, n_components): {(n_samples, n_components)}, "
        f"but got: {X_proj.shape}."
    )


@pytest.fixture
def structured_hierarchical_data_2d() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a dataset where X geometry matches the Y hierarchy. HierarchicalShape should excel here.
    """
    n_examples = 25
    y_list = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 7]]
    y = np.repeat(y_list, n_examples, axis=0).astype(float)

    level_1_offset = np.array([-50, 0]) * (1 - y[:, 0:1]) + np.array([50, 0]) * y[:, 0:1]
    level_2_offset = np.array([0, -10]) * (1 - y[:, 1:2]) + np.array([0, 10]) * y[:, 1:2]
    level_3_offset = np.random.randn(y.shape[0], 2) * 1.0
    X = level_1_offset + level_2_offset + level_3_offset

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]


def test_hierarchical_preserves_structure_in_2d(
    structured_hierarchical_data_2d: tuple[NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Tests that HierarchicalShape can be fit and produces reasonable scores.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y = structured_hierarchical_data_2d

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    assert X_proj.shape == (X.shape[0], smds_engine.stage_1.n_components), (
        f"Output shape should be (n_samples, n_components), but got {X_proj.shape}."
    )

    score = smds_engine.score(X, y)
    assert score > 0.9, f"The SMDS score should be high since X geometry matches Y hierarchy, but got {score:.4f}."


def test_hierarchical_distance_computation() -> None:
    """
    Tests that HierarchicalShape correctly computes distances based on the first differing level.
    """
    shape = HierarchicalShape(level_distances=np.array([1.0, 2.0, 3.0]))

    y = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 1, 1]]).astype(float)

    D = shape(y)

    assert D[0, 1] == 3.0, "Points differing at level 2 should have distance 3.0"
    assert D[0, 2] == 2.0, "Points differing at level 1 should have distance 2.0"
    assert D[0, 3] == 1.0, "Points differing at level 0 should have distance 1.0"
    assert D[0, 4] == 0.0, "Identical points should have distance 0.0"
    assert np.allclose(D, D.T), "Distance matrix should be symmetric"
    assert np.allclose(np.diag(D), 0), "Diagonal should be zero"


def test_hierarchical_validation() -> None:
    """
    Tests that HierarchicalShape correctly validates input dimensions.
    """
    shape = HierarchicalShape(level_distances=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="must be 2-dimensional"):
        shape(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="must have 2 columns"):
        shape(np.array([[1, 2, 3], [4, 5, 6]]))

    with pytest.raises(ValueError, match="cannot be empty"):
        shape(np.array([]).reshape(0, 2))

    valid_y = np.array([[1, 1], [1, 2], [2, 1]]).astype(float)
    D = shape(valid_y)
    assert D.shape == (3, 3), "Should accept valid 2D input"


def test_hierarchical_init_validation() -> None:
    """
    Tests that HierarchicalShape validates level_distances parameter.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        HierarchicalShape(level_distances=np.array([]))

    with pytest.raises(ValueError, match="must be non-negative"):
        HierarchicalShape(level_distances=np.array([1.0, -1.0, 2.0]))


@pytest.fixture
def structured_hierarchical_data_high_dim() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Provides a high-dimensional dataset where X geometry matches the Y hierarchy.
    """
    n_examples = 25
    y_list = [[0, 0, 0], [0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 4], [1, 0, 5], [1, 1, 6], [1, 1, 7]]
    y = np.repeat(y_list, n_examples, axis=0).astype(float)

    level_1_offset = np.array([-50, 0]) * (1 - y[:, 0:1]) + np.array([50, 0]) * y[:, 0:1]
    level_2_offset = np.array([0, -10]) * (1 - y[:, 1:2]) + np.array([0, 10]) * y[:, 1:2]
    level_3_offset = np.random.randn(y.shape[0], 2) * 1.0
    X_latent = level_1_offset + level_2_offset + level_3_offset

    high_dim = 10
    projection_matrix: NDArray[np.float64] = np.random.randn(2, high_dim)
    X_high_dim: NDArray[np.float64] = X_latent @ projection_matrix

    indices: NDArray[np.int_] = np.arange(X_high_dim.shape[0])
    np.random.shuffle(indices)

    return X_high_dim[indices], y[indices], X_latent[indices]


def test_hierarchical_recovers_structure_from_high_dim(
    structured_hierarchical_data_high_dim: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    smds_engine: SupervisedMDS,
) -> None:
    """
    Tests if SMDS can find and recover a hierarchical structure hidden in a noisy, high-dimensional space.
    """
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    X, y, _ = structured_hierarchical_data_high_dim

    X_proj: NDArray[np.float64] = smds_engine.fit_transform(X, y)

    assert X_proj.shape == (X.shape[0], smds_engine.stage_1.n_components), (
        f"Output shape should be (n_samples, n_components), but got {X_proj.shape}."
    )

    score = smds_engine.score(X, y)
    score_threshold = 0.9
    assert score > score_threshold, (
        f"The SMDS score is too low. Expected a score greater than {score_threshold}, but got {score:.4f}."
    )
