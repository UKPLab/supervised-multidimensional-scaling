import numpy as np
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

from smds.stress.kl_divergence import kl_divergence_stress

def test_kl_divergence_range() -> None:
    """
    Tests if the KL divergence is non-negative.
    KL(P||Q) must always be >= 0.
    """
    rng = np.random.default_rng(2137)
    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = squareform(pdist(X_high, metric="euclidean"))
    D_low = squareform(pdist(X_low, metric="euclidean"))

    kl_val = kl_divergence_stress(D_high, D_low, sigma=1.0)

    assert kl_val >= 0

def test_kl_divergence_perfect_preservation() -> None:
    """
    Tests if KL divergence is ~0 when the distances (distributions) are identical.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((15, 5))
    
    D = squareform(pdist(X, metric="euclidean"))

    kl_val = kl_divergence_stress(D, D, sigma=1.0)
    
    assert kl_val < 1e-10

def test_kl_divergence_scale_sensitivity() -> None:
    """
    Tests that KL divergence depends on the scale of the input distances
    (due to the Gaussian kernel width sigma being fixed relative to data scale).
    """
    rng = np.random.default_rng(1337)
    X_high = rng.standard_normal((15, 5))
    X_low = rng.standard_normal((15, 2))

    D_high = squareform(pdist(X_high, metric="euclidean"))
    D_low = squareform(pdist(X_low, metric="euclidean"))

    kl_original = kl_divergence_stress(D_high, D_low, sigma=1.0)

    D_low_scaled = D_low * 5.0
    kl_scaled = kl_divergence_stress(D_high, D_low_scaled, sigma=1.0)

    assert not np.isclose(kl_original, kl_scaled), \
        "KL Divergence (with fixed sigma) should be sensitive to scale."