import numpy as np
from scipy.spatial.distance import pdist

from smds.stress.scale_normalized_stress import ScaleNormalizedStress


def test_scale_normalized_stress_range() -> None:
    """
    Tests if the stress value is within the valid range [0, 1].
    """
    rng = np.random.default_rng(2137)
    X_high = rng.standard_normal((50, 5))
    X_low = X_high[:, :2]  # Simple projection

    # Compute distances first
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    sns = ScaleNormalizedStress()
    stress = sns.compute(D_high, D_low)

    assert 0 <= stress <= 1


def test_perfect_preservation() -> None:
    """
    Tests if stress is 0 when the distances are identical.
    """
    sns = ScaleNormalizedStress()
    rng = np.random.default_rng(42)
    
    X = rng.standard_normal((10, 5))
    
    # Compute distances (same for both since it's perfect preservation)
    D = pdist(X, metric="euclidean")
    
    stress = sns.compute(D, D)
    assert stress < 1e-10  # essentially 0


def test_scale_invariance() -> None:
    """
    Tests if the metric is truly scale-invariant.
    Scaling the low-dim distances by a factor should not change the stress.
    """
    sns = ScaleNormalizedStress()
    rng = np.random.default_rng(1337)

    X_high = rng.standard_normal((20, 5))
    X_low = rng.standard_normal((20, 2))

    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")

    # 1. Original stress
    stress_original = sns.compute(D_high, D_low)

    # 2. Scale up (zoom in)
    D_low_scaled = D_low * 1234.5
    stress_scaled = sns.compute(D_high, D_low_scaled)

    # 3. Scale down (zoom out)
    D_low_shrunk = D_low * 1e-5
    stress_shrunk = sns.compute(D_high, D_low_shrunk)

    np.testing.assert_allclose(stress_original, stress_scaled, err_msg="Stress changed after scaling up")
    np.testing.assert_allclose(stress_original, stress_shrunk, err_msg="Stress changed after scaling down")
