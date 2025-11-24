"""
Integration tests for the SupervisedMDS library.

This module validates that all implemented Shape classes can successfully
recover latent structures from high-dimensional noisy data.

NOTE ON SCORE THRESHOLDS:
Not all shapes can achieve a score of 1.0, even with perfect data.
This is due to fundamental geometric mismatches between the 'Ideal' shape definition
and the Euclidean embedding space:

Arc vs. Chord Mismatch (Geodesic, DiscreteCircular):
   These shapes define distance based on arcs (circle or steps around a ring).
   MDS embeds into a linear Euclidean space (chords). It is mathematically
   impossible to flatten a curved metric perfectly. Scores around 0.85-0.90
   represent the theoretical maximum for these topologies.
"""

import pytest
from scipy.spatial import procrustes
from smds import SupervisedMDS

# Format: (shape_name, engine_fixture_name, data_fixture_name, score_min, procrustes_max)

SHAPE_TEST_CASES = [
    ("Chain",           "chain_engine",         "chain_data_10d",           0.90, 0.2),
    ("Cluster",         "cluster_engine",       "cluster_data_10d",         0.90, 0.1),
    ("DiscCircular",    "disc_circular_engine", "disc_circular_data_10d",   0.70, 0.2),  # max score 0.87 without noise
    ("Hierarchical",    "hierarchical_engine",  "hierarchical_data_10d",    0.90, 0.1),
    ("Circular",        "circular_engine",      "circular_data_10d",        0.80, 0.1),
    ("Cylindrical",     "cylindrical_engine",   "cylindrical_data_10d",     0.80, 0.1),
    ("Spherical",       "spherical_engine",     "spherical_data_10d",       0.70, 0.2),
    ("Geodesic",        "geodesic_engine",      "geodesic_data_10d",        0.70, 0.2),  # max score 0.90 without noise
    ("Spiral",          "spiral_engine",        "spiral_data_10d",          0.90, 0.1),
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
    # Load fixtures
    smds_engine: SupervisedMDS = request.getfixturevalue(engine_name)
    data_tuple = request.getfixturevalue(data_name)

    # The data fixture returns 3 values, we only need X and y for a smoke test
    X_high, y, _ = data_tuple

    # Execution
    X_proj = smds_engine.fit_transform(X_high, y)

    # Assertion
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
        request: pytest.FixtureRequest
) -> None:
    """
    Universal Integration Test:
    Verifies that ANY shape engine can recover its specific latent structure
    from high-dimensional noisy data.
    """
    # Dynamically load the fixtures using the string names
    smds_engine: SupervisedMDS = request.getfixturevalue(engine_name)
    data_tuple = request.getfixturevalue(data_name)

    X_high, y, X_original = data_tuple

    # Execution
    X_proj = smds_engine.fit_transform(X_high, y)

    # Assertion: Procrustes
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)

    assert disparity < procrustes_max, (
        f"[{shape_name}] Shape recovery failed Procrustes analysis. "
        f"Disparity ({disparity:.4f}) exceeds threshold ({procrustes_max})."
    )

    # Assertion: Score
    score = smds_engine.score(X_high, y)
    assert score > score_min, (
        f"[{shape_name}] SMDS score is too low. "
        f"Expected > {score_min}, but got {score:.4f}."
    )
    print(f"\n{shape_name}Shape: Integration Test passed with \n - Score {score:.2f} \n - Disparity {disparity:.2f}\n")
