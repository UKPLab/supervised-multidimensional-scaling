import numpy as np

from smds.stress.scale_normalized_stress import scale_normalized_stress


def test_scale_normalized_stress_perfect_match() -> None:
    """Test that stress is 0 when distances match perfectly."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([1.0, 2.0, 3.0])
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-10


def test_scale_normalized_stress_zero_denominator_alpha() -> None:
    """Test behavior when all predicted distances are zero."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([0.0, 0.0, 0.0])
    stress = scale_normalized_stress(d_true, d_pred)
    assert np.isinf(stress)


def test_scale_normalized_stress_zero_denominator_d_true() -> None:
    """Test behavior when all true distances are zero."""
    d_true = np.array([0.0, 0.0, 0.0])
    d_pred = np.array([1.0, 2.0, 3.0])

    try:
        stress = scale_normalized_stress(d_true, d_pred)
        assert np.isinf(stress) or np.isnan(stress)
    except (ZeroDivisionError, ValueError):
        pass


def test_scale_normalized_stress_scale_invariance() -> None:
    """Test that scale-normalized stress IS scale invariant."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred_original = np.array([1.0, 2.0, 3.0])
    d_pred_scaled = np.array([10.0, 20.0, 30.0])  # Scaled by 10

    stress_original = scale_normalized_stress(d_true, d_pred_original)
    stress_scaled = scale_normalized_stress(d_true, d_pred_scaled)

    assert stress_original < 1e-10
    assert np.isclose(stress_original, stress_scaled, rtol=1e-5)


def test_scale_normalized_stress_proportional_relationship() -> None:
    """Test with proportional but not identical values."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([2.0, 4.0, 6.0])  # Exactly 2x
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-10


def test_scale_normalized_stress_single_element() -> None:
    """Test with single element array."""
    d_true = np.array([5.0])
    d_pred = np.array([3.0])
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-10


def test_scale_normalized_stress_opposite_correlation() -> None:
    """Test with negatively correlated distances."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([3.0, 2.0, 1.0])  # Reversed order
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress > 0.0


def test_scale_normalized_stress_large_values() -> None:
    """Test with very large values (numerical stability)."""
    d_true = np.array([1e10, 2e10])
    d_pred = np.array([1e10, 2e10])
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-5


def test_scale_normalized_stress_small_values() -> None:
    """Test with very small values (numerical stability)."""
    d_true = np.array([1e-10, 2e-10])
    d_pred = np.array([1e-10, 2e-10])
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-5


def test_scale_normalized_stress_negative_values() -> None:
    """Test with negative predicted values (edge case)."""
    d_true = np.array([1.0, 2.0])
    d_pred = np.array([-1.0, -2.0])
    stress = scale_normalized_stress(d_true, d_pred)
    assert stress < 1e-10

