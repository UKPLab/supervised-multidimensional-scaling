import numpy as np

from smds.stress.non_metric_stress import non_metric_stress


def test_non_metric_stress_perfect_match() -> None:
    """Test that stress is 0 when distances match perfectly."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([1.0, 2.0, 3.0])
    stress = non_metric_stress(d_true, d_pred)
    assert stress == 0.0


def test_non_metric_stress_scale_invariance() -> None:
    """
    Non-metric stress is scale invariant because it only cares about rank order.
    If d_pred maintains the same rank order as d_true, stress should be 0 regardless of scale.
    """
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred_original = np.array([1.0, 2.0, 3.0])
    d_pred_scaled = np.array([10.0, 20.0, 30.0])  # Scaled by 10, same rank order

    stress_original = non_metric_stress(d_true, d_pred_original)
    stress_scaled = non_metric_stress(d_true, d_pred_scaled)

    assert stress_original == 0.0
    assert stress_scaled == 0.0


def test_non_metric_stress_symmetry() -> None:
    """Test if metric handles inputs symmetrically if applicable (usually not symmetric in args)."""
    d1 = np.array([1.0, 2.0])
    d2 = np.array([2.0, 3.0])

    s1 = non_metric_stress(d1, d2)
    s2 = non_metric_stress(d2, d1)

    assert isinstance(s1, float)
    assert isinstance(s2, float)


def test_non_metric_stress_zero_denominator() -> None:
    """Test behavior with zero distances (potential division by zero)."""
    d_true = np.array([0.0, 0.0])
    d_pred = np.array([0.0, 0.0])

    stress = non_metric_stress(d_true, d_pred)
    assert np.isinf(stress)


def test_non_metric_stress_single_element() -> None:
    """Test with single element array."""
    d_true = np.array([5.0])
    d_pred = np.array([3.0])
    stress = non_metric_stress(d_true, d_pred)
    assert isinstance(stress, float)
    assert stress >= 0.0


def test_non_metric_stress_isotonic_monotonic() -> None:
    """Test that isotonic regression preserves monotonic relationships."""
    d_true = np.array([1.0, 2.0, 3.0, 4.0])
    d_pred = np.array([1.0, 2.0, 3.0, 4.0])
    stress = non_metric_stress(d_true, d_pred)
    assert stress == 0.0


def test_non_metric_stress_isotonic_non_monotonic() -> None:
    """Test isotonic regression with non-monotonic predictions."""
    d_true = np.array([1.0, 2.0, 3.0, 4.0])
    d_pred = np.array([4.0, 3.0, 2.0, 1.0])
    stress = non_metric_stress(d_true, d_pred)
    assert stress > 0.0


def test_non_metric_stress_ties() -> None:
    """Test with tied values."""
    d_true = np.array([1.0, 2.0, 2.0, 3.0])
    d_pred = np.array([1.0, 2.0, 2.0, 3.0])
    stress = non_metric_stress(d_true, d_pred)
    assert stress == 0.0


def test_non_metric_stress_negative_values() -> None:
    """Test with negative predicted values."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([-1.0, -2.0, -3.0])
    stress = non_metric_stress(d_true, d_pred)
    assert stress > 0.0


def test_non_metric_stress_large_differences() -> None:
    """Test with very large differences but different rank order."""
    d_true = np.array([1.0, 2.0])
    d_pred = np.array([1000.0, 100.0])  # Reversed order
    stress = non_metric_stress(d_true, d_pred)
    assert stress > 0.0


def test_non_metric_stress_small_values() -> None:
    """Test with very small values."""
    d_true = np.array([1e-10, 2e-10])
    d_pred = np.array([1e-10, 2e-10])
    stress = non_metric_stress(d_true, d_pred)
    assert stress == 0.0
