import numpy as np
from smds.stress.shepard_goodness_score import ShepardGoodnessScore


def test_shepard_goodness_perfect_preservation() -> None:
    sgs = ShepardGoodnessScore()

    X_high = np.array([[0], [1], [3]])
    X_low = np.array([[0], [1], [3]])

    score: float = sgs.compute(X_high, X_low)

    assert score > 0.99, f"Score should be close to 1, got {score}"


def test_shepard_goodness_scale_invariance() -> None:
    sgs = ShepardGoodnessScore()
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    score_original = sgs.compute(X_high, X_low)

    X_low_scaled = X_low * 555.0
    score_scaled = sgs.compute(X_high, X_low_scaled)

    X_low_shrunk = X_low * 0.0001
    score_shrunk = sgs.compute(X_high, X_low_shrunk)

    np.testing.assert_allclose(score_original, score_scaled, err_msg="Score changed after scaling up")
    np.testing.assert_allclose(score_original, score_shrunk, err_msg="Score changed after scaling down")


def test_shepard_goodness_range() -> None:
    sgs = ShepardGoodnessScore()
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))

    score: float = sgs.compute(X_high, X_low)

    assert -1 <= score <= 1