import os
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.model_selection import cross_validate  # type: ignore[import-untyped]

from smds import SupervisedMDS
from smds.shapes.base_shape import BaseShape
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.continuous_shapes.euclidean import EuclideanShape
from smds.shapes.continuous_shapes.log_linear import LogLinearShape
from smds.shapes.continuous_shapes.semicircular import SemicircularShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape
from smds.shapes.spatial_shapes.cylindrical import CylindricalShape
from smds.shapes.spatial_shapes.geodesic import GeodesicShape
from smds.shapes.spatial_shapes.spherical import SphericalShape
from smds.shapes.spiral_shape import SpiralShape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_results")

DEFAULT_SHAPES = [
    ChainShape(),
    ClusterShape(),
    DiscreteCircularShape(),
    CircularShape(),
    CylindricalShape(),
    GeodesicShape(),
    SphericalShape(),
    SpiralShape(),
    LogLinearShape(),
    EuclideanShape(),
    SemicircularShape(),
]


def discover_manifolds(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    shapes: Optional[List[BaseShape]] = None,
    n_folds: int = 5,
    n_jobs: int = -1,
    save_results: bool = True,
    save_path: Optional[str] = None,
    experiment_name: str = "results",
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Evaluates a list of Shape hypotheses on the given data using Cross-Validation.

    Args:
        X: High-dimensional data (n_samples, n_features).
        y: Labels (n_samples,).
        shapes: List of Shape objects to test. Defaults to a standard set if None.
        n_folds: Number of Cross-Validation folds.
        n_jobs: Number of parallel jobs for cross_validate (-1 = all CPUs).
        save_results: Whether to persist results to a CSV file.
        save_path: Specific path to save results. If None, generates one based on timestamp.
        experiment_name: Label to include in the generated filename.

    Returns:
        A tuple containing:
        - pd.DataFrame: The aggregated results, sorted by mean score.
        - Optional[str]: The path to the saved CSV file, or None if saving was disabled.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES

    results_list = []

    # Configure persistence and resume logic
    if save_results:
        os.makedirs(SAVE_DIR, exist_ok=True)

        if save_path is None:
            # Create a unique, descriptive filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]  # Short 6-char unique hash

            # Clean experiment name (remove spaces/slashes)
            safe_name = "".join(c for c in experiment_name if c.isalnum() or c in ("-", "_"))

            filename = f"{safe_name}_{timestamp}_{unique_id}.csv"
            save_path = os.path.join(SAVE_DIR, filename)

        # Create file with header if it doesn't exist
        if not os.path.exists(save_path):
            pd.DataFrame(
                columns=[
                    "shape",
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "fold_scores",
                    "error",
                ]
            ).to_csv(save_path, index=False)

    print("Saving to:", save_path)

    # Filter shapes based on input dimension compatibility
    user_y_ndim = np.asarray(y).ndim

    valid_shapes = [s for s in shapes if s.y_ndim == user_y_ndim]

    skipped = len(shapes) - len(valid_shapes)
    if skipped > 0:
        print(
            f"Filtering: Kept {len(valid_shapes)} shapes, "
            f"skipped {skipped} due to dimension mismatch (Expected {user_y_ndim}D)."
        )

    for shape in valid_shapes:
        shape_name = shape.__class__.__name__

        estimator = SupervisedMDS(n_components=2, manifold=shape)

        # Run Cross-Validation
        # This handles splitting, training, testing, and scoring automatically.
        try:
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=n_folds,
                n_jobs=n_jobs,
                scoring=None,  # Uses estimator.score() by default
                return_train_score=False,
            )

            mean_score = np.mean(cv_results["test_score"])
            std_score = np.std(cv_results["test_score"])

            row = {
                "shape": shape_name,
                "params": shape.__dict__,
                "mean_test_score": mean_score,
                "std_test_score": std_score,
                "fold_scores": cv_results["test_score"],
                "error": None,
            }

        except ValueError as e:
            # Data Mismatch (e.g. 1D y vs 2D Shape)
            # This is expected behavior when running "all shapes" on specific data.
            print(f"Skipping {shape_name}: Incompatible Data Format. ({str(e)})")
            continue

        except Exception as e:
            # Unexpected Crash
            print(f"Skipping {shape_name}: {e}")
            row = {
                "shape": shape_name,
                "params": shape.__dict__,
                "mean_test_score": np.nan,
                "std_test_score": np.nan,
                "fold_scores": None,
                "error": str(e),
            }

        results_list.append(row)

        # Incrementally write result to disk
        if save_results and save_path is not None:
            pd.DataFrame([row]).to_csv(
                save_path,
                mode="a",
                header=False,
                index=False,
            )

    df = pd.DataFrame(results_list)

    if not df.empty and "mean_test_score" in df.columns:
        df = df.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    return df, save_path
