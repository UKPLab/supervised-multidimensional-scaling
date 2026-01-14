import ast
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

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
)-> Tuple[Dict[str, pd.DataFrame], Path]:
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

    # Pre-load all CSVs to avoid re-reading files for every metric
    loaded_dfs = []
    for run_idx, csv_path in enumerate(pipeline_results_paths):
        try:
            df = pd.read_csv(csv_path)
            loaded_dfs.append((run_idx, df))
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    pivot_dfs = {}  # Store pivots to return
    full_summary_report = {}

    # Iterate over ALL metrics defined in the Enum
    for metric in StressMetrics:
        target_metric = metric.value
        col_fold = f"fold_{target_metric}"

        print(f"Analyzing metric: {target_metric}...")

        data_records = []

        for run_idx, df in loaded_dfs:
            for _, row in df.iterrows():
                shape_name = row['shape']

                if col_fold not in row:
                    continue

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

        if not data_records:
            print(f"No valid data found for metric {target_metric}")
            continue

        agg_df = pd.DataFrame(data_records)

        # Pivot to get Shapes as columns
        # Index will be (run_id, fold_id)
        pivot_df = agg_df.pivot(index=["run_id", "fold_id"], columns="shape", values="score")
        pivot_dfs[target_metric] = pivot_df

        # Perform Analysis and capture summary
        print(f"Analyzing {target_metric}...")
        metric_summary = perform_st_analysis(pivot_df, st_output_dir, target_metric)

        # Add to master report
        full_summary_report[target_metric] = metric_summary

    # Save Aggregated Summary JSON
    summary_path = os.path.join(st_output_dir, "st_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary_report, f, indent=4)

    print(f"Statistical validation complete. Results saved in {st_output_dir}")

    return pivot_dfs, Path(st_output_dir)
