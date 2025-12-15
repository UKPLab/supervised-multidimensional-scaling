import pytest
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# We will implement this function later
from smds.discovery_pipline import discover_manifolds

# Import Shapes to test
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape
from smds.shapes.discrete_shapes.hierarchical import HierarchicalShape
from smds.shapes.base_shape import BaseShape


# =============================================================================
# TEST CASES
# =============================================================================

def test_pipeline_returns_dataframe(cluster_data_10d: tuple[NDArray, NDArray, NDArray]):
    """Smoke test: Does it return a dataframe with correct columns?"""
    X, y, _ = cluster_data_10d
    shapes = [ClusterShape(), ChainShape()]

    results = discover_manifolds(X, y, shapes=shapes, n_folds=2)

    assert isinstance(results, pd.DataFrame)
    expected_cols = ["shape", "params", "mean_test_score", "std_test_score"]
    # Check that all expected columns are present
    assert all(col in results.columns for col in expected_cols)
    # Check that we have one row per shape
    assert len(results) == 2


def test_cluster_wins_on_cluster_data(cluster_data_10d: tuple[NDArray, NDArray, NDArray]):
    """Logic test: Does ClusterShape win on Cluster Data?"""
    X, y, _ = cluster_data_10d

    # We test a few shapes against each other
    shapes = [
        ClusterShape(),
        ChainShape(threshold=1.1),
        CircularShape(radious=1.0)
    ]

    results = discover_manifolds(X, y, shapes=shapes, n_folds=5)

    # Sort by score descending
    results = results.sort_values("mean_test_score", ascending=False)

    # The winner should be ClusterShape
    winner = results.iloc[0]["shape"]
    assert winner == "ClusterShape"

    # The score should be high (> 0.9)
    assert results.iloc[0]["mean_test_score"] > 0.9


def test_circular_wins_on_circular_data(circular_data_10d: tuple[NDArray, NDArray, NDArray]):
    """Logic test: Does CircularShape win on Circular Data?"""
    X, y, _ = circular_data_10d

    results = discover_manifolds(X, y, n_folds=5)
    results = results.sort_values("mean_test_score", ascending=False)

    winner = results.iloc[0]["shape"]

    # Assert Circular wins
    assert winner == "CircularShape"