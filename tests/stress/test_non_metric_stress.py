import numpy as np
from scipy.spatial.distance import pdist

from smds.stress.non_metric_stress import NonMetricStress


def test_non_metric_stress_perfect_monotonic_preservation() -> None:
    """
    Tests if stress is effectively zero when the rank order of distances is perfectly preserved.
    """
    nms = NonMetricStress()

    # 1. High-Dim: 3 points on a line: 0, 1, 3 -> Distances: 1, 2, 3
    X_high = np.array([[0], [1], [3]])

    # 2. Low-Dim: Distortion preserving order -> Distances: 10, 90, 100
    X_low_distorted = np.array([[0], [10], [100]])

    # IMPORTANT: Compute pairwise distances first!
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low_distorted, metric="euclidean")

    stress: float = nms.compute(D_high, D_low)

    assert stress < 1e-9, f"Stress should be 0, got {stress}"


def test_non_metric_stress_scale_invariance() -> None:
    """
    Tests if non-metric stress is invariant to uniform scaling (zooming).
    """
    nms = NonMetricStress()

    rng = np.random.default_rng(42)
    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    # Compute base distances
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress_original = nms.compute(D_high, D_low)

    # Scale the DISTANCES, simulating a zoomed plot
    D_low_scaled = D_low * 555.0
    stress_scaled = nms.compute(D_high, D_low_scaled)

    D_low_shrunk = D_low * 0.0001
    stress_shrunk = nms.compute(D_high, D_low_shrunk)

    np.testing.assert_allclose(stress_original, stress_scaled, err_msg="Stress changed after scaling up")
    np.testing.assert_allclose(stress_original, stress_shrunk, err_msg="Stress changed after scaling down")


def test_non_metric_stress_range() -> None:
    """
    Verifies that the computed stress value lies within the valid range [0, 1].
    """
    nms = NonMetricStress()
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))

    # Compute distances
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress: float = nms.compute(D_high, D_low)

    assert 0 <= stress <= 1
    assert stress > 0.01