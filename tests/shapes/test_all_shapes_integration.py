import pytest
from scipy.spatial import procrustes
from numpy.typing import NDArray
import numpy as np
from smds import SupervisedMDS

# =============================================================================
# TEST CONFIGURATION (The "Menu")
# =============================================================================
# Format: (shape_name, engine_fixture_name, data_fixture_name, score_min, procrustes_max)

SHAPE_TEST_CASES = [
    ("Chain",          "chain_engine",         "chain_data_10d",         0.90, 0.2),
    ("Cluster",        "cluster_engine",       "cluster_data_10d",       0.90, 0.1),
    ("DiscCircular",   "disc_circular_engine", "disc_circular_data_10d", 0.70, 0.2),
    ("Hierarchical",   "hierarchical_engine",  "hierarchical_data_10d",  0.90, 0.1),
    ("Circular",       "circular_engine",      "circular_data_10d",      0.80, 0.1),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "shape_name, engine_name, data_name, score_min, procrustes_max",
    SHAPE_TEST_CASES
)
def test_shape_smoke_execution(
        shape_name: str,
        engine_name: str,
        data_name: str,
        score_min: float,
        procrustes_max: float,
        request: pytest.FixtureRequest
) -> None:
    """
    Universal Smoke Test:
    Ensures that EVERY shape engine can be fit and transformed without crashing,
    and produces an output of the correct shape.
    """
    # 1. Load fixtures
    smds_engine: SupervisedMDS = request.getfixturevalue(engine_name)
    data_tuple = request.getfixturevalue(data_name)

    # The data fixture returns 3 values, we only need X and y for a smoke test
    X_high, y, _ = data_tuple

    # 2. Execution
    X_proj = smds_engine.fit_transform(X_high, y)

    # 3. Assertion
    n_samples = X_high.shape[0]
    n_components = smds_engine.n_components

    assert X_proj.shape == (n_samples, n_components), (
        f"[{shape_name}] Output shape is incorrect. "
        f"Expected {(n_samples, n_components)}, but got {X_proj.shape}."
    )
    print(f"\n{shape_name}Shape: Smoke Test passed\n")

# =============================================================================
# THE UNIFIED TEST FUNCTION
# =============================================================================

@pytest.mark.parametrize(
    "shape_name, engine_name, data_name, score_min, procrustes_max",
    SHAPE_TEST_CASES
)
def test_shape_recovers_structure_from_high_dim(
        shape_name: str,
        engine_name: str,
        data_name: str,
        score_min: float,
        procrustes_max: float,
        request: pytest.FixtureRequest  # <--- The key to loading fixtures by name
) -> None:
    """
    Universal Integration Test:
    Verifies that ANY shape engine can recover its specific latent structure
    from high-dimensional noisy data.
    """
    # 1. Dynamically load the fixtures using the string names
    smds_engine: SupervisedMDS = request.getfixturevalue(engine_name)
    data_tuple = request.getfixturevalue(data_name)

    X_high, y, X_original = data_tuple

    # 2. Execution

    X_proj = smds_engine.fit_transform(X_high, y)

    # 3. Assertion: Procrustes (Shape Fidelity)
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)

    assert disparity < procrustes_max, (
        f"[{shape_name}] Shape recovery failed Procrustes analysis. "
        f"Disparity ({disparity:.4f}) exceeds threshold ({procrustes_max})."
    )

    # 4. Assertion: Score (Internal Consistency)
    score = smds_engine.score(X_high, y)
    assert score > score_min, (
        f"[{shape_name}] SMDS score is too low. "
        f"Expected > {score_min}, but got {score:.4f}."
    )
    print(f"\n{shape_name}Shape: Integration Test passed with Score {score:.2f}\n")