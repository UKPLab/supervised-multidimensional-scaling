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
from scipy.spatial import procrustes  # type: ignore[import-untyped]

from smds import SupervisedMDS

# Format: (shape_name, engine_fixture_name, data_fixture_name, score_min, procrustes_max)

SHAPE_TEST_CASES = [
    ("Chain_Computed", "chain_engine_computed", "chain_data_10d", 0.90, 0.2),
    ("Chain_UserProvided", "chain_engine_user_provided", "chain_data_10d", 0.85, 0.2),
    ("Cluster_Computed", "cluster_engine_computed", "cluster_data_10d", 0.80, 0.2),
    ("Cluster_UserProvided", "cluster_engine_user_provided", "cluster_data_10d", 0.80, 0.2),
    (
        "DiscCircular_Computed",
        "disc_circular_engine_computed",
        "disc_circular_data_10d",
        0.70,
        0.2,
    ),  # max score 0.87 without noise
    (
        "DiscCircular_UserProvided",
        "disc_circular_engine_user_provided",
        "disc_circular_data_10d",
        0.70,
        0.2,
    ),
    ("Hierarchical_Computed", "hierarchical_engine_computed", "hierarchical_data_10d", 0.90, 0.1),
    ("Hierarchical_UserProvided", "hierarchical_engine_user_provided", "hierarchical_data_10d", 0.90, 0.1),
    ("Circular_Computed", "circular_engine_computed", "circular_data_10d", 0.80, 0.1),
    ("Circular_UserProvided", "circular_engine_user_provided", "circular_data_10d", 0.80, 0.1),
    ("Cylindrical_Computed", "cylindrical_engine_computed", "cylindrical_data_10d", 0.80, 0.1),
    ("Cylindrical_UserProvided", "cylindrical_engine_user_provided", "cylindrical_data_10d", 0.80, 0.1),
    ("Spherical_Computed", "spherical_engine_computed", "spherical_data_10d", 0.70, 0.2),
    ("Spherical_UserProvided", "spherical_engine_user_provided", "spherical_data_10d", 0.70, 0.2),
    (
        "Geodesic_Computed",
        "geodesic_engine_computed",
        "spherical_data_10d",
        0.70,
        0.2,
    ),  # max score 0.90 without noise
    ("Geodesic_UserProvided", "geodesic_engine_user_provided", "spherical_data_10d", 0.70, 0.2),
    ("Spiral_Computed", "spiral_engine_computed", "spiral_data_10d", 0.90, 0.1),
    ("Spiral_UserProvided", "spiral_engine_user_provided", "spiral_data_10d", 0.90, 0.1),
    ("LogLinear_Computed", "log_linear_engine_computed", "log_linear_data_10d", 0.70, 0.2),
    ("LogLinear_UserProvided", "log_linear_engine_user_provided", "log_linear_data_10d", 0.70, 0.2),
    ("Euclidean_Computed", "euclidean_engine_computed", "euclidean_data_10d", 0.90, 0.1),
    ("Euclidean_UserProvided", "euclidean_engine_user_provided", "euclidean_data_10d", 0.90, 0.1),
    ("Semicircular_Computed", "semicircular_engine_computed", "semicircular_data_10d", 0.70, 0.2),
    ("Semicircular_UserProvided", "semicircular_engine_user_provided", "semicircular_data_10d", 0.70, 0.2),
    ("KleinBottle_Computed", "klein_bottle_engine_computed", "klein_bottle_data_10d", 0.70, 0.2),
    ("KleinBottle_UserProvided", "klein_bottle_engine_user_provided", "klein_bottle_data_10d", 0.70, 0.2),
    ("Torus", "torus_engine", "torus_data_10d", 0.65, 0.2),
    ("GraphGeodesic_Computed", "graph_geodesic_engine_computed", "graph_geodesic_data_10d", 0.35, 0.95),
    (
        "GraphGeodesic_UserProvided",
        "graph_geodesic_engine_user_provided",
        "graph_geodesic_data_10d",
        0.35,
        0.95,
    ),
    ("Polytope_Computed", "polytope_engine_computed", "polytope_data_10d", 0.35, 0.99),
    ("Polytope_UserProvided", "polytope_engine_user_provided", "polytope_data_10d", 0.35, 0.99),
]


@pytest.mark.smoke
@pytest.mark.parametrize("shape_name, engine_name, data_name, score_min, procrustes_max", SHAPE_TEST_CASES)
def test_shape_smoke_execution(
    shape_name: str,
    engine_name: str,
    data_name: str,
    score_min: float,
    procrustes_max: float,
    request: pytest.FixtureRequest,
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
    X_high, y, x_original = data_tuple
    y_for_fit = x_original if "UserProvided" in shape_name else y

    # Execution
    X_proj = smds_engine.fit_transform(X_high, y_for_fit)

    # Assertion
    n_samples = X_high.shape[0]
    n_components = smds_engine.parametrization_fitted_.n_components

    assert X_proj.shape == (n_samples, n_components), (
        f"[{shape_name}] Output shape is incorrect. Expected {(n_samples, n_components)}, but got {X_proj.shape}."
    )
    print(f"\n{shape_name}Shape: Smoke Test passed\n")


# =============================================================================
# THE UNIFIED TEST FUNCTION
# =============================================================================


@pytest.mark.parametrize("shape_name, engine_name, data_name, score_min, procrustes_max", SHAPE_TEST_CASES)
def test_shape_recovers_structure_from_high_dim(
    shape_name: str,
    engine_name: str,
    data_name: str,
    score_min: float,
    procrustes_max: float,
    request: pytest.FixtureRequest,
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
    y_for_fit = X_original if "UserProvided" in shape_name else y

    # Execution
    X_proj = smds_engine.fit_transform(X_high, y_for_fit)

    # Assertion: Procrustes
    mtx1, mtx2, disparity = procrustes(X_original, X_proj)

    assert disparity < procrustes_max, (
        f"[{shape_name}] Shape recovery failed Procrustes analysis. "
        f"Disparity ({disparity:.4f}) exceeds threshold ({procrustes_max})."
    )

    # Assertion: Score
    score = smds_engine.score(X_high, y_for_fit)
    assert score > score_min, f"[{shape_name}] SMDS score is too low. Expected > {score_min}, but got {score:.4f}."
    print(f"\n{shape_name}Shape: Integration Test passed with \n - Score {score:.2f} \n - Disparity {disparity:.2f}\n")
