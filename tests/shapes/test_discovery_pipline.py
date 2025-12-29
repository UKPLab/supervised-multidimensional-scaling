"""
Tests for the Manifold Discovery Pipeline logic.
Verifies that the pipeline correctly orchestrates cross-validation,
sorts results, and identifies the correct winning shape for known data topologies.
"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from numpy.typing import NDArray

from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.spiral_shape import SpiralShape


@pytest.mark.smoke
def test_pipeline_returns_dataframe(
    cluster_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Smoke test: Verify pipeline returns a valid DataFrame with expected columns."""
    X, y, _ = cluster_data_10d
    shapes = [ClusterShape(), CircularShape()]

    results, _ = discover_manifolds(
        X,
        y,
        shapes=shapes,
        n_folds=2,
        experiment_name="Smoke_Test",
        save_results=False,
        clear_cache=True
    )

    assert isinstance(results, pd.DataFrame)
    expected_cols = ["shape", "params", "mean_test_score", "std_test_score"]
    # Check schema
    assert all(col in results.columns for col in expected_cols)
    # Check row count
    assert len(results) == 2


def test_cluster_wins_on_cluster_data(
    cluster_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Logic test: Verify ClusterShape achieves the highest score on Cluster Data."""
    X, y, _ = cluster_data_10d

    # We test a list of shapes
    shapes = [
        ClusterShape(),
        CircularShape(),
        SpiralShape(),
    ]

    results, _ = discover_manifolds(
        X,
        y,
        shapes=shapes,
        n_folds=5,
        experiment_name="Cluster_Test",
        save_results=False,
        clear_cache=True
    )

    # Sort by score descending
    results = results.sort_values("mean_test_score", ascending=False)

    # The winner should be ClusterShape
    winner = results.iloc[0]["shape"]
    assert winner == "ClusterShape"


def test_circular_wins_on_circular_data(
    circular_data_10d: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Logic test: Verify CircularShape wins on Circular Data (using default shapes list)."""
    X, y, _ = circular_data_10d

    # Using default shapes list (shapes=None)
    results, _ = discover_manifolds(
        X,
        y,
        n_folds=5,
        experiment_name="Circular_Test",
        save_results=False,
        clear_cache=True
    )

    results = results.sort_values("mean_test_score", ascending=False)

    winner = results.iloc[0]["shape"]
    assert winner == "CircularShape"
