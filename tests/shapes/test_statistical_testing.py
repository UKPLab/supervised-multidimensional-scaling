import json
import numpy as np
import pytest
from smds.pipeline.statistical_testing.run_statistical_test import run_statistical_validation
from smds.stress.stress_metrics import StressMetrics


@pytest.mark.slow
def test_st_pipeline_with_random_data(tmp_path):
    """
    Validates the architecture of the ST pipeline (File I/O, Aggregation).
    Uses random noise to ensure the pipeline handles 'messy' data without crashing,
    and produces the expected file artifacts.
    """

    rng = np.random.default_rng(42)
    X = rng.random((100, 3))
    y = rng.random(100)

    pivot_dfs, output_dir = run_statistical_validation(
        X, y,
        n_repeats=5,
        n_folds=5,
        experiment_name="test_random_data"
    )

    # A. Check Root Directory
    assert output_dir.exists()
    assert (output_dir / "st_summary.json").exists()

    # B. Check Summary Content
    with open(output_dir / "st_summary.json", "r") as f:
        summary = json.load(f)

    # Ensure our target metric is in the summary
    target_metric = StressMetrics.SCALE_NORMALIZED_STRESS.value
    assert target_metric in summary
    assert "p_value" in summary[target_metric]
    assert "statistic" in summary[target_metric]

    # C. Check Subfolder Generation
    # Even if results aren't significant, the folder might be created or not depending on logic.
    # But if significant (which is likely with bias), check for CSVs.
    metric_dir = output_dir / target_metric
    if summary[target_metric]['significant']:
        assert metric_dir.exists()
        assert (metric_dir / "p_values.csv").exists()
        assert (metric_dir / "cd_diagram.png").exists()


@pytest.mark.slow
def test_st_pipeline_with_circular(circular_data_10d, tmp_path):
    """
    Validates the Statistical Logic.
    When given highly structured data (Circular), the Friedman test
    MUST return a significant P-Value (< 0.05).
    """

    X, y, _ = circular_data_10d

    _, output_dir = run_statistical_validation(
        X, y,
        n_repeats=5,
        n_folds=5,
        experiment_name="test_circular_data"
    )

    with open(output_dir / "st_summary.json", "r") as f:
        summary = json.load(f)

    target_metric = StressMetrics.SCALE_NORMALIZED_STRESS.value
    stats = summary[target_metric]

    print(f"\n[Structure Test] Friedman P-Value: {stats['p_value']}")

    # The pipeline MUST verify that the shapes performed differently
    assert stats['significant'] is True
    assert stats['p_value'] < 0.05

    # Ensure the statistic (Separation Score) is positive
    assert stats['statistic'] > 1.0