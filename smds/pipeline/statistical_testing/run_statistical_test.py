import ast
import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.pipeline.statistical_testing.analysis import perform_st_analysis
from smds.stress.stress_metrics import StressMetrics


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST_RESULTS_DIR = os.path.join(BASE_DIR, "statistical_testing", "st_results")


def run_statistical_validation(
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        n_folds: int = 5,
        experiment_name: str = "st_experiment"
):
    """
    Runs the discovery pipeline `n_repeats` times with different random seeds.
    Aggregates results and performs statistical analysis.
    """

    # Generate Unique ST ID
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    st_run_id = f"st_run_{experiment_name}_{timestamp}_{unique_id}"

    st_output_dir = os.path.join(ST_RESULTS_DIR, st_run_id)
    os.makedirs(st_output_dir, exist_ok=True)

    print(f"Starting Statistical Validation: {st_run_id}")
    print(f"Repeats: {n_repeats}, Folds: {n_folds}")

    pipeline_results_paths = []

    # Loop Repeats
    for i in range(n_repeats):
        seed = 42 + i * 1337  # Deterministic seeds for reproducibility of the whole ST run

        print(f"--- Run {i + 1}/{n_repeats} (Seed: {seed}) ---")

        _, csv_path = discover_manifolds(
            X, y,
            n_folds=n_folds,
            save_results=True,
            experiment_name=f"{experiment_name}_rep{i + 1}",
            random_state=seed,
            st_run_id=st_run_id,
            clear_cache=True
        )

        if csv_path:
            pipeline_results_paths.append(csv_path)

    # Aggregate Data
    print("Aggregating results...")

    # todo: For now we only focus on Scale Normalized Stress for the analysis. Ticket expects all metrics
    target_metric = StressMetrics.SCALE_NORMALIZED_STRESS.value
    col_mean = f"mean_{target_metric}"
    col_fold = f"fold_{target_metric}"

    # Matrix: Rows = (Run_i, Fold_j), Cols = Shapes
    data_records = []

    for run_idx, csv_path in enumerate(pipeline_results_paths):
        # We need to skip the metadata comment lines when reading for aggregation
        df = pd.read_csv(csv_path, comment='#')

        for _, row in df.iterrows():
            shape_name = row['shape']
            # Parse the string representation of list back to list
            try:
                fold_scores = ast.literal_eval(row[col_fold])
                if not isinstance(fold_scores, list):
                    continue
            except:
                continue

            for fold_idx, score in enumerate(fold_scores):
                data_records.append({
                    "run_id": run_idx,
                    "fold_id": fold_idx,
                    "shape": shape_name,
                    "score": score
                })

    agg_df = pd.DataFrame(data_records)

    # Pivot to get Shapes as columns
    # Index will be (run_id, fold_id)
    pivot_df = agg_df.pivot(index=["run_id", "fold_id"], columns="shape", values="score")

    # Perform Analysis
    print("Performing Friedman/Nemenyi tests...")
    perform_st_analysis(pivot_df, st_output_dir, target_metric)

    print(f"Statistical validation complete. Results saved in {st_output_dir}")

    return pivot_df, Path(st_output_dir)
