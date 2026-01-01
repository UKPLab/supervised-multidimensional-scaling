import numpy as np

from smds.stress.shepard_goodness_score import shepard_goodness_stress


def test_shepard_perfect_correlation() -> None:
    """Test perfect positive correlation (monotonic increasing)."""
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    score = shepard_goodness_stress(d_true, d_pred)
    assert np.isclose(score, 1.0)


def test_shepard_perfect_correlation_nonlinear() -> None:
    """Test perfect rank correlation even with non-linear values."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([1.0, 100.0, 1000.0])  # Monotonically increasing
    score = shepard_goodness_stress(d_true, d_pred)
    assert score == 1.0


def test_shepard_perfect_negative_correlation() -> None:
    """Test perfect negative correlation."""
    d_true = np.array([1.0, 2.0, 3.0, 4.0])
    d_pred = np.array([40.0, 30.0, 20.0, 10.0])
    score = shepard_goodness_stress(d_true, d_pred)
    assert score == -1.0


def test_shepard_no_correlation() -> None:
    """Test case with no correlation."""
    d_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_pred = np.array([3.0, 5.0, 1.0, 4.0, 2.0])
    score = shepard_goodness_stress(d_true, d_pred)
    assert np.isclose(score, -0.3)


def test_shepard_identical_inputs() -> None:
    """Test when inputs are identical."""
    d = np.random.default_rng(42).random(10)
    score = shepard_goodness_stress(d, d)
    assert np.isclose(score, 1.0)


def test_shepard_ties_in_data() -> None:
    """Test handling of tied values (same ranks)."""
    d_true = np.array([1.0, 2.0, 2.0, 3.0])
    d_pred = np.array([1.0, 2.0, 2.0, 3.0])
    score = shepard_goodness_stress(d_true, d_pred)
    assert score == 1.0


def test_shepard_single_element() -> None:
    """Test with single element array."""
    d_true = np.array([5.0])
    d_pred = np.array([3.0])
    score = shepard_goodness_stress(d_true, d_pred)
    assert np.isnan(score) or score == 1.0


def test_shepard_all_zeros() -> None:
    """Test with all zero values."""
    d_true = np.array([0.0, 0.0, 0.0])
    d_pred = np.array([0.0, 0.0, 0.0])
    try:
        score = shepard_goodness_stress(d_true, d_pred)
        assert np.isnan(score) or score == 1.0
    except ValueError:
        pass


def test_shepard_constant_one_array() -> None:
    """Test when one array is constant."""
    d_true = np.array([1.0, 2.0, 3.0])
    d_pred = np.array([5.0, 5.0, 5.0])
    try:
        score = shepard_goodness_stress(d_true, d_pred)
        assert np.isnan(score)
    except ValueError:
        pass


def test_shepard_very_large_values() -> None:
    """Test with very large values."""
    d_true = np.array([1e10, 2e10, 3e10])
    d_pred = np.array([1e10, 2e10, 3e10])
    score = shepard_goodness_stress(d_true, d_pred)
    assert score == 1.0


def test_shepard_very_small_values() -> None:
    """Test with very small values."""
    d_true = np.array([1e-10, 2e-10, 3e-10])
    d_pred = np.array([1e-10, 2e-10, 3e-10])
    score = shepard_goodness_stress(d_true, d_pred)
    assert score == 1.0
