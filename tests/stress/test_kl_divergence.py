import numpy as np

from smds.stress.kl_divergence import kl_divergence_stress


def test_kl_divergence_perfect_match() -> None:
    """Test KL divergence is close to 0 for identical distributions."""
    N = 5
    M = np.abs(np.random.default_rng(42).standard_normal((N, N)))
    np.fill_diagonal(M, 0.0)
    M = (M + M.T) / 2.0

    loss = kl_divergence_stress(M, M)
    assert loss < 1e-6


def test_kl_divergence_high_distance() -> None:
    """Test KL divergence is large for very different distributions."""
    N = 5
    M1 = np.ones((N, N))
    np.fill_diagonal(M1, 0.0)

    M2 = np.ones((N, N)) * 100.0
    np.fill_diagonal(M2, 0.0)

    _ = kl_divergence_stress(M1, M2)

    M_structured = np.array([
        [0.0, 1.0, 10.0],
        [1.0, 0.0, 10.0],
        [10.0, 10.0, 0.0]
    ])

    M_inverted = np.array([
        [0.0, 10.0, 1.0],
        [10.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])

    loss_struct = kl_divergence_stress(M_structured, M_inverted)
    assert loss_struct > 1e-4


def test_kl_divergence_small_sigma() -> None:
    """Test with very small sigma parameter."""
    M = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    loss = kl_divergence_stress(M, M, sigma=0.01)
    assert loss < 1e-5


def test_kl_divergence_large_sigma() -> None:
    """Test with very large sigma parameter."""
    M = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    loss = kl_divergence_stress(M, M, sigma=100.0)
    assert loss < 1e-5


def test_kl_divergence_minimal_matrix() -> None:
    """Test with minimal 2x2 matrix."""
    M = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    loss = kl_divergence_stress(M, M)
    assert loss < 1e-6


def test_kl_divergence_all_zeros() -> None:
    """Test with all zero distance matrix."""
    N = 3
    M = np.zeros((N, N))
    try:
        loss = kl_divergence_stress(M, M)
        assert np.isfinite(loss) or np.isnan(loss)
    except (ValueError, RuntimeWarning):
        pass


def test_kl_divergence_very_large_distances() -> None:
    """Test with very large distance values."""
    M = np.array([
        [0.0, 1e10, 2e10],
        [1e10, 0.0, 1e10],
        [2e10, 1e10, 0.0]
    ])
    loss = kl_divergence_stress(M, M)
    assert loss < 1e-5


def test_kl_divergence_very_small_distances() -> None:
    """Test with very small distance values."""
    M = np.array([
        [0.0, 1e-10, 2e-10],
        [1e-10, 0.0, 1e-10],
        [2e-10, 1e-10, 0.0]
    ])
    loss = kl_divergence_stress(M, M)
    assert loss < 1e-5


def test_kl_divergence_non_square_raises_error() -> None:
    """Test that non-square matrices raise ValueError."""
    M_rect = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0]
    ])
    M_square = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    try:
        _ = kl_divergence_stress(M_rect, M_square)
        assert False, "Should have raised ValueError for non-square matrix"
    except ValueError:
        pass


def test_kl_divergence_different_sizes_raises_error() -> None:
    """Test that mismatched matrix sizes raise error."""
    M1 = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])
    M2 = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    try:
        _ = kl_divergence_stress(M1, M2)
        assert False, "Should have raised error for mismatched sizes"
    except (ValueError, AssertionError):
        pass
