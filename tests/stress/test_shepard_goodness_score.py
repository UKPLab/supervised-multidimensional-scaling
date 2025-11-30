import numpy as np
from scipy.spatial.distance import pdist

from smds.stress.shepard_goodness_score import shepard_goodness_stress


def test_shepard_goodness_perfect_preservation() -> None:
    X_high = np.array([[0], [1], [3]])
    X_low = np.array([[0], [1], [3]])

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    score: float = shepard_goodness_stress(D_high, D_low)

    assert score > 0.99, f"Score should be close to 1, got {score}"


def test_shepard_goodness_scale_invariance() -> None:
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    score_original = shepard_goodness_stress(D_high, D_low)

    D_low_scaled = D_low * 555.0
    score_scaled = shepard_goodness_stress(D_high, D_low_scaled)

    D_low_shrunk = D_low * 0.0001
    score_shrunk = shepard_goodness_stress(D_high, D_low_shrunk)

    np.testing.assert_allclose(score_original, score_scaled, err_msg="Score changed after scaling up")
    np.testing.assert_allclose(score_original, score_shrunk, err_msg="Score changed after scaling down")


def test_shepard_goodness_range() -> None:
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    score: float = shepard_goodness_stress(D_high, D_low)

    assert -1 <= score <= 1