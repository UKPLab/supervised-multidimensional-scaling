import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from sklearn.model_selection import cross_validate  # type: ignore[import-untyped]

from smds import SupervisedMDS
from smds.shapes.base_shape import BaseShape

# Default shapes list (lazy import or define here)
# For MVP, we can require the user to pass shapes or define a small default set
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
        n_jobs: int = -1
) -> pd.DataFrame:
    """
    Evaluates a list of Shape hypotheses on the given data using Cross-Validation.

    Args:
        X: High-dimensional data (n_samples, n_features).
        y: Labels (n_samples,).
        shapes: List of Shape objects to test. Defaults to a standard set.
        n_folds: Number of Cross-Validation folds.
        n_jobs: Number of parallel jobs for cross_validate (-1 = all CPUs).

    Returns:
        pd.DataFrame containing the results, sorted by mean score.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES

    results_list = []

    # Detect User Input Dimension
    user_y_ndim = np.asarray(y).ndim

    # Filter list of shapes before we start
    valid_shapes = [
        s for s in shapes
        if s.y_ndim == user_y_ndim
    ]

    # Inform the user
    skipped = len(shapes) - len(valid_shapes)
    if skipped > 0:
        print(f"Filtering: Kept {len(valid_shapes)} shapes, "
              f"skipped {skipped} due to dimension mismatch (Expected {user_y_ndim}D).")

    for shape in valid_shapes:
        shape_name = shape.__class__.__name__

        # 1. Create the Estimator
        # We wrap the shape in the Engine
        estimator = SupervisedMDS(n_components=2, manifold=shape)

        # 2. Run Cross-Validation
        # This handles splitting, training, testing, and scoring automatically.
        try:
            cv_results = cross_validate(
                estimator, X, y,
                cv=n_folds,
                n_jobs=n_jobs,
                scoring=None,  # Uses estimator.score() by default
                return_train_score=False
            )

            # 3. Aggregate Results
            mean_score = np.mean(cv_results['test_score'])
            std_score = np.std(cv_results['test_score'])

            results_list.append({
                "shape": shape_name,
                "params": shape.__dict__,  # Captures config like threshold=1.1
                "mean_test_score": mean_score,
                "std_test_score": std_score,
                "fold_scores": cv_results['test_score']  # Optional: keep raw scores
            })

        except ValueError as e:

            # Data Mismatch (e.g. 1D y vs 2D Shape)
            # This is expected behavior when running "all shapes" on specific data.
            print(f"Skipping {shape_name}: Incompatible Data Format. ({str(e)})")

        except Exception as e:
            # Unexpected Crash (Algorithm Bug)
            print(f"Skipping {shape_name}: {e}")
            results_list.append({
                "shape": shape_name,
                "params": shape.__dict__,
                "mean_test_score": np.nan,
                "std_test_score": np.nan,
                "error": str(e)
            })

    # 4. Create DataFrame
    df = pd.DataFrame(results_list)

    # Sort by best score
    if not df.empty and "mean_test_score" in df.columns:
        df = df.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    #print(df.to_string())

    return df
