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
    ("Chain_ComputedStage1", "chain_engine_computed_stage1", "chain_data_10d", 0.90, 0.2),
    ("Chain_UserProvidedStage1", "chain_engine_user_provided_stage1", "chain_data_10d", 0.85, 0.2),
    ("Cluster_ComputedStage1", "cluster_engine_computed_stage1", "cluster_data_10d", 0.80, 0.2),
    ("Cluster_UserProvidedStage1", "cluster_engine_user_provided_stage1", "cluster_data_10d", 0.80, 0.2),
    (
        "DiscCircular_ComputedStage1",
        "disc_circular_engine_computed_stage1",
        "disc_circular_data_10d",
        0.70,
        0.2,
    ),  # max score 0.87 without noise
    (
        "DiscCircular_UserProvidedStage1",
        "disc_circular_engine_user_provided_stage1",
        "disc_circular_data_10d",
        0.70,
        0.2,
    ),
    ("Hierarchical_ComputedStage1", "hierarchical_engine_computed_stage1", "hierarchical_data_10d", 0.90, 0.1),
    ("Hierarchical_UserProvidedStage1", "hierarchical_engine_user_provided_stage1", "hierarchical_data_10d", 0.90, 0.1),
    ("Circular_ComputedStage1", "circular_engine_computed_stage1", "circular_data_10d", 0.80, 0.1),
    ("Circular_UserProvidedStage1", "circular_engine_user_provided_stage1", "circular_data_10d", 0.80, 0.1),
    ("Cylindrical_ComputedStage1", "cylindrical_engine_computed_stage1", "cylindrical_data_10d", 0.80, 0.1),
    ("Cylindrical_UserProvidedStage1", "cylindrical_engine_user_provided_stage1", "cylindrical_data_10d", 0.80, 0.1),
    ("Spherical_ComputedStage1", "spherical_engine_computed_stage1", "spherical_data_10d", 0.70, 0.2),
    ("Spherical_UserProvidedStage1", "spherical_engine_user_provided_stage1", "spherical_data_10d", 0.70, 0.2),
    (
        "Geodesic_ComputedStage1",
        "geodesic_engine_computed_stage1",
        "spherical_data_10d",
        0.70,
        0.2,
    ),  # max score 0.90 without noise
    ("Geodesic_UserProvidedStage1", "geodesic_engine_user_provided_stage1", "spherical_data_10d", 0.70, 0.2),
    ("Spiral_ComputedStage1", "spiral_engine_computed_stage1", "spiral_data_10d", 0.90, 0.1),
    ("Spiral_UserProvidedStage1", "spiral_engine_user_provided_stage1", "spiral_data_10d", 0.90, 0.1),
    ("LogLinear_ComputedStage1", "log_linear_engine_computed_stage1", "log_linear_data_10d", 0.70, 0.2),
    ("LogLinear_UserProvidedStage1", "log_linear_engine_user_provided_stage1", "log_linear_data_10d", 0.70, 0.2),
    ("Euclidean_ComputedStage1", "euclidean_engine_computed_stage1", "euclidean_data_10d", 0.90, 0.1),
    ("Euclidean_UserProvidedStage1", "euclidean_engine_user_provided_stage1", "euclidean_data_10d", 0.90, 0.1),
    ("Semicircular_ComputedStage1", "semicircular_engine_computed_stage1", "semicircular_data_10d", 0.70, 0.2),
    ("Semicircular_UserProvidedStage1", "semicircular_engine_user_provided_stage1", "semicircular_data_10d", 0.70, 0.2),
    ("KleinBottle_ComputedStage1", "klein_bottle_engine_computed_stage1", "klein_bottle_data_10d", 0.70, 0.2),
    ("KleinBottle_UserProvidedStage1", "klein_bottle_engine_user_provided_stage1", "klein_bottle_data_10d", 0.70, 0.2),
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
    y_for_fit = x_original if "UserProvidedStage1" in shape_name else y

    # Execution
    X_proj = smds_engine.fit_transform(X_high, y_for_fit)

    # Assertion
    n_samples = X_high.shape[0]
    n_components = smds_engine.stage_1_fitted_.n_components

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
    y_for_fit = X_original if "UserProvidedStage1" in shape_name else y

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
