import numpy as np

from smds.stress.non_metric_stress import NonMetricStress


def test_non_metric_stress_perfect_monotonic_preservation() -> None:
    """
    Tests if stress is effectively zero when the rank order of distances is perfectly preserved,
    even if the absolute distances are heavily distorted non-linearly.
    """
    nms = NonMetricStress()

    X_high = np.array([[0], [1], [3]])

    X_low_distorted = np.array([[0], [10], [1000]])

    stress: float = nms.compute(X_high, X_low_distorted)

    assert stress < 1e-9


def test_non_metric_stress_scale_invariance() -> None:
    """
    Tests if non-metric stress is invariant to uniform scaling (zooming) of the embedding.
    """
    nms = NonMetricStress()

    rng = np.random.default_rng(42)
    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    stress_original = nms.compute(X_high, X_low)

    X_low_scaled = X_low * 555.0
    stress_scaled = nms.compute(X_high, X_low_scaled)

    X_low_shrunk = X_low * 0.0001
    stress_shrunk = nms.compute(X_high, X_low_shrunk)

    np.testing.assert_allclose(stress_original, stress_scaled)
    np.testing.assert_allclose(stress_original, stress_shrunk)


def test_non_metric_stress_range() -> None:
    """
    Verifies that the computed stress value lies within the valid range [0, 1].
    """
    nms = NonMetricStress()
    rng = np.random.default_rng(2137)

    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))

    stress: float = nms.compute(X_high, X_low)

    assert 0 <= stress <= 1
    assert stress > 0.01
