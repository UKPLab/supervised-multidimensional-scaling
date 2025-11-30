"""
Integration tests for all stress metrics in the SupervisedMDS library.

This module validates that all implemented stress metrics work correctly
across different distance ranges and produce valid outputs. It specifically
handles the differences between vector-based metrics (Stress, Shepard) and
matrix-based metrics (KL Divergence).

Test Coverage:
- Original range: Tests with original distance values.
- Scaled range: Tests with scaled distances to verify scale invariance properties.
- Perfect preservation: Validates output when D_high equals D_low.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

from smds.stress.kl_divergence import kl_divergence_stress
from smds.stress.non_metric_stress import non_metric_stress
from smds.stress.normalized_stress import normalized_stress
from smds.stress.scale_normalized_stress import scale_normalized_stress
from smds.stress.shepard_goodness_score import shepard_goodness_stress
from smds.stress.stress_metrics import StressMetrics

STRESS_TEST_CASES = [
    (
        StressMetrics.SCALE_NORMALIZED_STRESS,
        scale_normalized_stress,
        (0.0, 1.0),
        True,
        False,
    ),
    (
        StressMetrics.NON_METRIC_STRESS,
        non_metric_stress,
        (0.0, 1.0),
        True,
        False,
    ),
    (
        StressMetrics.NORMALIZED_STRESS,
        normalized_stress,
        (0.0, np.inf),
        False,
        False,
    ),
    (
        StressMetrics.SHEPARD_GOODNESS_SCORE,
        shepard_goodness_stress,
        (-1.0, 1.0),
        True,
        False,
    ),
    (
        StressMetrics.NORMALIZED_KL_DIVERGENCE,
        kl_divergence_stress,
        (0.0, np.inf),
        False,
        True,
    ),
]


def _to_matrix(D_flat: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Converts a flattened distance array (from pdist) to a symmetric square matrix.
    Required for metrics like KL Divergence that calculate local probabilities.
    """
    return squareform(D_flat)  # type: ignore[no-any-return]


def _invoke_metric(
    func: Callable[..., float],
    is_matrix_based: bool,
    D_true: NDArray[np.float64],
    D_pred: NDArray[np.float64],
    **kwargs: float
) -> float:
    """
    Helper to invoke the metric function, handling data format differences.

    Args:
        func: The metric function to call.
        is_matrix_based: If True, converts inputs to square matrices.
        D_true: Flattened ground truth distances.
        D_pred: Flattened predicted distances.
        **kwargs: Additional arguments for the metric (e.g., sigma).
    """
    if is_matrix_based:
        M_true = _to_matrix(D_true)
        M_pred = _to_matrix(D_pred)
        return float(func(M_true, M_pred, **kwargs))
    
    return float(func(D_true, D_pred))


@pytest.fixture
def original_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generates synthetic high-dimensional and low-dimensional distance vectors."""
    rng = np.random.default_rng(42)
    X_high = rng.standard_normal((50, 10))
    X_low = rng.standard_normal((50, 2))
    D_high = pdist(X_high, metric="euclidean")
    D_low = pdist(X_low, metric="euclidean")
    return D_high, D_low


@pytest.fixture
def scaled_data(
    original_data: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Returns the original data, but with the low-dim distances scaled up significantly."""
    D_high, D_low = original_data
    scale_factor = 1234.5
    D_low_scaled = D_low * scale_factor
    return D_high, D_low_scaled


@pytest.fixture
def perfect_preservation_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generates a case where D_high is identical to D_low (perfect embedding)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    D = pdist(X, metric="euclidean")
    return D, D.copy()


@pytest.mark.parametrize(
    "metric_enum, metric_func, expected_range, is_scale_invariant, is_matrix_based",
    STRESS_TEST_CASES,
)
def test_stress_original_range(
    metric_enum: StressMetrics,
    metric_func: Callable[..., float],
    expected_range: tuple[float, float],
    is_scale_invariant: bool,
    is_matrix_based: bool,
    original_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    Tests if the metric returns a value within the theoretically valid range
    for standard random data.
    """
    D_high, D_low = original_data
    result = _invoke_metric(metric_func, is_matrix_based, D_high, D_low, sigma=1.0)
    
    min_val, max_val = expected_range
    assert min_val <= result <= max_val, (
        f"[{metric_enum.value}] Result {result} outside expected range {expected_range}"
    )


@pytest.mark.parametrize(
    "metric_enum, metric_func, expected_range, is_scale_invariant, is_matrix_based",
    STRESS_TEST_CASES,
)
def test_stress_scaled_range(
    metric_enum: StressMetrics,
    metric_func: Callable[..., float],
    expected_range: tuple[float, float],
    is_scale_invariant: bool,
    is_matrix_based: bool,
    original_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    scaled_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    Tests the scale invariance property.
    - If invariant: Original and Scaled results must be (nearly) identical.
    - If not invariant: Scaled result must still be valid numbers within range.
    """
    D_high, D_low = original_data
    _, D_low_scaled = scaled_data

    result_original = _invoke_metric(metric_func, is_matrix_based, D_high, D_low, sigma=1.0)
    result_scaled = _invoke_metric(metric_func, is_matrix_based, D_high, D_low_scaled, sigma=1.0)

    if is_scale_invariant:
        np.testing.assert_allclose(
            result_original, 
            result_scaled, 
            rtol=1e-5,
            err_msg=f"[{metric_enum.value}] Metric failed scale invariance check."
        )
    else:
        min_val, max_val = expected_range
        assert min_val <= result_scaled <= max_val, (
            f"[{metric_enum.value}] Scaled result {result_scaled} outside expected range."
        )
        assert not np.isclose(result_original, result_scaled), (
            f"[{metric_enum.value}] Metric should be scale-sensitive, "
            f"but got identical values: {result_original:.6f} == {result_scaled:.6f}"
        )


@pytest.mark.parametrize(
    "metric_enum, metric_func, expected_range, is_scale_invariant, is_matrix_based",
    STRESS_TEST_CASES,
)
def test_stress_score_range_perfect_preservation(
    metric_enum: StressMetrics,
    metric_func: Callable[..., float],
    expected_range: tuple[float, float], 
    is_scale_invariant: bool,  
    is_matrix_based: bool,
    perfect_preservation_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    Tests the metric behavior when the embedding is perfect (D_high == D_low).
    Expects 0.0 for stress/divergence metrics and 1.0 for correlation metrics.
    """
    D_high, D_low = perfect_preservation_data
    result = _invoke_metric(metric_func, is_matrix_based, D_high, D_low, sigma=1.0)

    if metric_enum == StressMetrics.SHEPARD_GOODNESS_SCORE:
        assert result > 0.999
    elif metric_enum == StressMetrics.NORMALIZED_KL_DIVERGENCE:
        assert result < 1e-5
    else:
        assert result < 1e-5