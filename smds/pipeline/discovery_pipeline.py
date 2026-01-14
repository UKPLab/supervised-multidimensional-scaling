import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.model_selection import cross_validate, KFold  # type: ignore[import-untyped]

from smds import SupervisedMDS
from smds.shapes.base_shape import BaseShape
from smds.shapes.continuous_shapes import (
    CircularShape,
    EuclideanShape,
    KleinBottleShape,
    LogLinearShape,
    SemicircularShape,
    SpiralShape,
)
from smds.shapes.discrete_shapes import ChainShape, ClusterShape, DiscreteCircularShape
from smds.shapes.spatial_shapes import CylindricalShape, GeodesicShape, SphericalShape
from smds.stress.stress_metrics import StressMetrics

from .helpers.hash import compute_shape_hash, hash_data, load_cached_shape_result, save_shape_result
from .helpers.interactive_plots import generate_interactive_plot
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
    KleinBottleShape(),
]


def discover_manifolds(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    shapes: Optional[List[BaseShape]] = None,
    smds_components: int = 2,
    n_folds: int = 5,
    n_jobs: int = -1,
    save_results: bool = True,
    experiment_name: str = "results",
    create_png_visualization: bool = True,
    clear_cache: bool = True,
    random_state: Optional[int] = None,
    st_run_id: Optional[str] = None,
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
        smds_components: Tells SMDS on how many dimensions to map
        shapes: List of Shape objects to test. Defaults to a standard set if None.
        n_folds: Number of Cross-Validation folds. If 0, Cross-Validation is skipped and
                 the model is fit and scored directly on all data.
        n_jobs: Number of parallel jobs for cross_validate (-1 = all CPUs).
        save_results: Whether to persist results to a CSV file.
        experiment_name: Label to include in the generated filename.
        create_png_visualization: Whether to create a visualization of the results as an image file.
        clear_cache: Whether to delete all cache files after successful completion.
        random_state: Seed used by the random number generator for Cross-Validation splitting.
        st_run_id: Unique identifier for a Statistical Testing run.

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

    if st_run_id is not None:
        # to avoid loding cached manifolds from prev pipeline_run in st_run
        clear_cache = True

    results_list = []

    data_hash = hash_data(X, y)
    os.makedirs(CACHE_DIR, exist_ok=True)

    metric_columns = []
    for m in StressMetrics:
        metric_columns.extend([f"mean_{m.value}", f"std_{m.value}", f"fold_{m.value}"])

    csv_headers = ["shape", "params"] + metric_columns + ["error", "plot_path"]

    experiment_dir = None
    plots_dir = None
    unique_suffix = ""

    if save_results:
        os.makedirs(SAVE_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]

        safe_name = "".join(c for c in experiment_name if c.isalnum() or c in ("-", "_"))

        # Create a unique folder for this experiment
        unique_suffix = f"{safe_name}_{timestamp}_{unique_id}"
        experiment_dir = os.path.join(SAVE_DIR, unique_suffix)
        plots_dir = os.path.join(experiment_dir, "plots")

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        filename = f"{unique_suffix}.csv"
        save_path = os.path.join(experiment_dir, filename)

        # Save Metadata to JSON
        # used to let the dashboard know which experiments belong to which st_run
        meta_path = os.path.join(experiment_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"st_run_id": st_run_id}, f, indent=4)

        with open(save_path, 'w', encoding='utf-8') as f:
            pd.DataFrame(columns=csv_headers).to_csv(save_path, index=False)

        print("Saving to:", save_path)

    # Setup Cross-Validation Strategy
    cv_strategy: Any

    # If a seed is provided (ST run), we MUST build the splitter manually to inject the seed.
    if random_state is not None:
        cv_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # If no seed (Standard run), keep original behavior (let sklearn handle it via int)
    else:
        cv_strategy = n_folds

    # Construct the scoring map for cross_validate

    # Define a type alias for clarity: (estimator, X, y) -> float
    ScorerFunc = Callable[[Any, Any, Any], float]
    scoring_map: Dict[str, ScorerFunc] = {}

    for metric in StressMetrics:

        def make_scorer(m: StressMetrics) -> ScorerFunc:
            # estimator.score returns a float
            return lambda estimator, x_data, y_data: float(estimator.score(x_data, y_data, metric=m))

        scoring_map[metric.value] = make_scorer(metric)

    # Filter shapes based on input dimension compatibility
    user_y_ndim = np.asarray(y).ndim

    valid_shapes = [s for s in shapes if s.y_ndim == user_y_ndim]

    skipped = len(shapes) - len(valid_shapes)
    if skipped > 0:
        print(
            f"Filtering: Kept {len(valid_shapes)} shapes, "
            f"skipped {skipped} due to dimension mismatch (Expected {user_y_ndim}D)."
        )

    if X.shape[0] < 100:
        print("[WARNING] Less than 100 datapoints in X might lead to noisy results")

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

        estimator = SupervisedMDS(n_components=smds_components, manifold=shape)

        try:
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=cv_strategy,
                n_jobs=n_jobs,
                scoring=scoring_map,
                return_train_score=False,
            )

            row = {
                "shape": shape_name,
                "params": params,
            }

            for metric in StressMetrics:
                metric_key = metric.value
                cv_key = f"test_{metric_key}"

                if cv_key in cv_results:
                    scores = cv_results[cv_key]
                    row[f"mean_{metric_key}"] = np.mean(scores)
                    row[f"std_{metric_key}"] = np.std(scores)
                    row[f"fold_{metric_key}"] = scores.tolist()
                else:
                    row[f"mean_{metric_key}"] = np.nan

            row["error"] = None

            # Generate Interactive Plot
            if save_results and plots_dir is not None:
                try:
                    full_estimator = SupervisedMDS(n_components=smds_components, manifold=shape)
                    X_embedded = full_estimator.fit_transform(X, y)

                    plot_name_prefix = f"{shape_name}_{unique_suffix}" if unique_suffix else shape_name

                    plot_filename = generate_interactive_plot(
                        X_embedded=X_embedded, y=y, shape_name=plot_name_prefix, save_dir=plots_dir
                    )

                    row["plot_path"] = os.path.join("plots", plot_filename)

                except Exception as plot_e:
                    print(f"Warning: Failed to generate interactive plot for {shape_name}: {plot_e}")
                    row["plot_path"] = None
            else:
                row["plot_path"] = None

            save_shape_result(cache_file, row)
            print(f"Computed and cached {shape_name}")

        except ValueError as e:
            print(f"Skipping {shape_name}: Incompatible Data Format. ({str(e)})")
            continue

        except Exception as e:
            print(f"Skipping {shape_name}: {e}")
            row = {"shape": shape_name, "params": params}
            for m in StressMetrics:
                row[f"mean_{m.value}"] = np.nan
                row[f"std_{m.value}"] = np.nan
                row[f"fold_{m.value}"] = []

            row["error"] = str(e)
            row["plot_path"] = None

            save_shape_result(cache_file, row)

        results_list.append(row)

        if save_results and save_path is not None:
            pd.DataFrame([row], columns=csv_headers).to_csv(
                save_path,
                mode="a",
                header=False,
                index=False,
            )

    df = pd.DataFrame(results_list)

    # Sort by the primary metric
    primary_metric = f"mean_{StressMetrics.SCALE_NORMALIZED_STRESS.value}"

    if not df.empty and primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False).reset_index(drop=True)

    if save_results and save_path is not None and create_png_visualization:
        create_plots(X, y, df, valid_shapes, save_path, experiment_name)

    if clear_cache:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("Cache cleared")

    return df, save_path
