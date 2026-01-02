import numpy as np

from smds.stress import normalized_stress


def test_normalized_stress_perfect_match() -> None:
    """Test that stress is 0 when distances match perfectly."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([1.0, 2.0, 3.0])
    stress = normalized_stress(d_true, d_pred)
    assert stress == 0.0


def test_normalized_stress_zero_denominator() -> None:
    """Test behavior when all true distances are zero (division by zero)."""
    d_true = np.array([0.0, 0.0, 0.0])
    d_pred = np.array([1.0, 2.0, 3.0])
    stress = normalized_stress(d_true, d_pred)
    assert np.isinf(stress)


def test_normalized_stress_scale_sensitive() -> None:
    """Test that normalized stress is NOT scale invariant."""
    d_true = np.array([1.0, 2.0])
    d_pred_original = np.array([1.0, 2.0])
    d_pred_scaled = np.array([10.0, 20.0])  # Scaled by 10

    stress_original = normalized_stress(d_true, d_pred_original)
    stress_scaled = normalized_stress(d_true, d_pred_scaled)

    assert stress_original == 0.0
    assert stress_scaled > 0.0
    assert not np.isclose(stress_original, stress_scaled)


def test_normalized_stress_opposite_values() -> None:
    """Test with opposite sign values (if distances can be negative)."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([-1.0, -2.0, -3.0])  # Opposite signs
    stress = normalized_stress(d_true, d_pred)
    assert stress > 0.0


def test_normalized_stress_single_element() -> None:
    """Test with single element array."""
    d_true = np.array([5.0])
    d_pred = np.array([3.0])
    stress = normalized_stress(d_true, d_pred)
    expected_squared = ((3.0 - 5.0) ** 2) / (5.0**2)
    expected = np.sqrt(expected_squared)
    assert np.isclose(stress, expected)


def test_normalized_stress_large_differences() -> None:
    """Test with very large differences."""
    d_true = np.array([1.0, 1.0])
    d_pred = np.array([1000.0, 1000.0])
    stress = normalized_stress(d_true, d_pred)
    assert stress > 100.0


def test_normalized_stress_small_values() -> None:
    """Test with very small values (numerical stability)."""
    d_true = np.array([1e-10, 2e-10])
    d_pred = np.array([1e-10, 2e-10])
    stress = normalized_stress(d_true, d_pred)
    assert stress == 0.0


def test_normalized_stress_mixed_signs() -> None:
    """Test with mixed positive and negative values in prediction."""
    d_true = np.array([1.0, 2.0])
    d_pred = np.array([1.0, -2.0])  # One negative
    stress = normalized_stress(d_true, d_pred)
    assert stress > 0.0
