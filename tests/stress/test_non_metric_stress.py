import numpy as np
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from smds.stress.non_metric_stress import non_metric_stress


def test_non_metric_stress_perfect_monotonic_preservation() -> None:
    """
    Tests if stress is effectively zero when the rank order of distances is perfectly preserved.
    """
    X_high = np.array([[0], [1], [3]])

    X_low_distorted = np.array([[0], [10], [100]])

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low_distorted, metric="euclidean")

    stress: float = non_metric_stress(D_high, D_low)

    assert stress < 1e-9, f"Stress should be 0, got {stress}"


def test_non_metric_stress_scale_invariance() -> None:
    """
    Tests if non-metric stress is invariant to uniform scaling (zooming).
    """
    rng = np.random.default_rng(42)
    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress_original = non_metric_stress(D_high, D_low)

    D_low_scaled = D_low * 555.0
    stress_scaled = non_metric_stress(D_high, D_low_scaled)

    D_low_shrunk = D_low * 0.0001
    stress_shrunk = non_metric_stress(D_high, D_low_shrunk)

    np.testing.assert_allclose(stress_original, stress_scaled, err_msg="Stress changed after scaling up")
    np.testing.assert_allclose(stress_original, stress_shrunk, err_msg="Stress changed after scaling down")


def test_non_metric_stress_range() -> None:
    """
    Verifies that the computed stress value lies within the valid range [0, 1].
    """
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress: float = non_metric_stress(D_high, D_low)

    assert 0 <= stress <= 1
    assert stress > 0.01
