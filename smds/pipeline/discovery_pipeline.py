import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
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

from .helpers.hash import compute_shape_hash, hash_data, load_cached_shape_result, save_shape_result
from .helpers.plots import create_plots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_results")
CACHE_DIR = os.path.join(SAVE_DIR, ".cache")

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
    create_visualization: bool = True,
    clear_cache: bool = False,
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Evaluates a list of Shape hypotheses on the given data using Cross-Validation or direct scoring.

    Features caching mechanism: Completed shapes are cached and can be recovered after
    a pipeline crash. Each shape's results are hashed based on data, shape parameters,
    and fold configuration. If the pipeline crashes, re-running with the same parameters
    will load cached results instead of recomputing.

    Args:
        X: High-dimensional data (n_samples, n_features).
        y: Labels (n_samples,).
        shapes: List of Shape objects to test. Defaults to a standard set if None.
        n_folds: Number of Cross-Validation folds. If 0, Cross-Validation is skipped and
                 the model is fit and scored directly on all data.
        n_jobs: Number of parallel jobs for cross_validate (-1 = all CPUs).
        save_results: Whether to persist results to a CSV file.
        save_path: Specific path to save results. If None, generates one based on timestamp.
        experiment_name: Label to include in the generated filename.
        create_visualization: Whether to create a visualization of the results as an image file.
        clear_cache: Whether to delete all cache files after successful completion.

    Returns:
        A tuple containing:
        - pd.DataFrame: The aggregated results, sorted by mean score.
        - Optional[str]: The path to the saved CSV file, or None if saving was disabled.

    Note:
        Cache files are stored in saved_results/.cache/ and are automatically used
        when re-running the pipeline with identical data and parameters.
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES

    results_list = []

    data_hash = hash_data(X, y)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if save_results:
        os.makedirs(SAVE_DIR, exist_ok=True)

        if save_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]

            safe_name = "".join(c for c in experiment_name if c.isalnum() or c in ("-", "_"))

            filename = f"{safe_name}_{timestamp}_{unique_id}.csv"
            save_path = os.path.join(SAVE_DIR, filename)

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
        params = shape.__dict__

        shape_hash = compute_shape_hash(shape_name, params, data_hash, n_folds)
        cache_file = os.path.join(CACHE_DIR, f"{shape_hash}.pkl")

        cached_result = load_cached_shape_result(cache_file)
        if cached_result is not None:
            print(f"Loading cached result for {shape_name}")
            row = cached_result
            results_list.append(row)
            continue

        estimator = SupervisedMDS(n_components=2, manifold=shape)

        try:
            if n_folds == 0:
                estimator.fit(X, y)
                score = estimator.score(X, y)
                mean_score = score
                std_score = 0.0
                fold_scores = [score]
            else:
                cv_results = cross_validate(
                    estimator,
                    X,
                    y,
                    cv=n_folds,
                    n_jobs=n_jobs,
                    scoring=None,
                    return_train_score=False,
                )

                mean_score = np.mean(cv_results["test_score"])
                std_score = np.std(cv_results["test_score"])
                fold_scores = cv_results["test_score"].tolist()

            row = {
                "shape": shape_name,
                "params": params,
                "mean_test_score": mean_score,
                "std_test_score": std_score,
                "fold_scores": fold_scores,
                "error": None,
            }

            save_shape_result(cache_file, row)
            print(f"Computed and cached {shape_name}")

        except ValueError as e:
            print(f"Skipping {shape_name}: Incompatible Data Format. ({str(e)})")
            continue

        except Exception as e:
            print(f"Skipping {shape_name}: {e}")
            row = {
                "shape": shape_name,
                "params": params,
                "mean_test_score": np.nan,
                "std_test_score": np.nan,
                "fold_scores": None,
                "error": str(e),
            }

            save_shape_result(cache_file, row)

        results_list.append(row)

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

    if save_results and save_path is not None and create_visualization:
        create_plots(X, y, df, valid_shapes, save_path, experiment_name)

    if clear_cache:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print(f"Cache cleared: {CACHE_DIR}")

    return df, save_path
