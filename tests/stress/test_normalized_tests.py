import numpy as np
from scipy.spatial.distance import pdist
from smds.stress.normalized_stress import normalized_stress

def test_normalized_stress_range() -> None:
    """
    Tests if the stress value is non-negative.
    Note: Normalized Stress is not bounded by 1.0, but must be >= 0.
    """
    rng = np.random.default_rng(2137)
    X_high = rng.standard_normal((50, 5))
    X_low = X_high[:, :2]

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress = normalized_stress(D_high, D_low)

    assert stress >= 0

def test_normalized_stress_perfect_preservation() -> None:
    """
    Tests if stress is 0 when the distances are identical.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 5))
    D = pdist(X, metric="euclidean")

    stress = normalized_stress(D, D)
    
    assert stress < 1e-10

def test_normalized_stress_scale_sensitivity() -> None:
    """
    Tests if the metric is scale-sensitive (as described in Smelser et al. 2025).
    Scaling the low-dim distances SHOULD change the stress significantly.
    """
    rng = np.random.default_rng(1337)
    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    stress_original = normalized_stress(D_high, D_low)

    D_low_scaled = D_low * 10.0
    stress_scaled = normalized_stress(D_high, D_low_scaled)

    assert not np.isclose(stress_original, stress_scaled), \
        "Normalized Stress should change with scale, but it remained constant."
    
    assert stress_scaled > stress_original