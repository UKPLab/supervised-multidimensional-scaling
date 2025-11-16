import numpy as np

from smds.stress.scale_normalized_stress import ScaleNormalizedStress


def test_scale_normalized_stress_perfect_embedding() -> None:
    np.random.seed(2137)
    X_high = np.random.randn(2137, 5)
    X_low = X_high[:, :2]

    sns = ScaleNormalizedStress()
    stress = sns.compute(X_high, X_low)

    assert 0 <= stress <= 1


def test_perfect_preservation() -> None:
    sns = ScaleNormalizedStress()
    X = np.random.randn(10, 5)
    stress = sns.compute(X, X)
    assert stress < 1e-10  # essentially 0
