import numpy as np
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from smds.stress.scale_normalized_stress import scale_normalized_stress


def test_scale_normalized_stress_range() -> None:
    """
    Tests if the stress value is within the valid range [0, 1].
    """
    rng = np.random.default_rng(2137)
    X_high = rng.standard_normal((50, 5))
    X_low = X_high[:, :2]

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress = scale_normalized_stress(D_high, D_low)

    assert 0 <= stress <= 1


def test_perfect_preservation() -> None:
    """
    Tests if stress is 0 when the distances are identical.
    """
    rng = np.random.default_rng(42)

    X = rng.standard_normal((10, 5))

    D = pdist(X, metric="euclidean")

    stress = scale_normalized_stress(D, D)
    assert stress < 1e-10


def test_scale_invariance() -> None:
    """
    Tests if the metric is truly scale-invariant.
    Scaling the low-dim distances by a factor should not change the stress.
    """
    rng = np.random.default_rng(1337)

    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress_original = scale_normalized_stress(D_high, D_low)

    D_low_scaled = D_low * 1234.5
    stress_scaled = scale_normalized_stress(D_high, D_low_scaled)

    D_low_shrunk = D_low * 1e-5
    stress_shrunk = scale_normalized_stress(D_high, D_low_shrunk)

    np.testing.assert_allclose(stress_original, stress_scaled, err_msg="Stress changed after scaling up")
    np.testing.assert_allclose(stress_original, stress_shrunk, err_msg="Stress changed after scaling down")
