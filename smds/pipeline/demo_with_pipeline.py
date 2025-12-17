"""
Demo script for Supervised Multi-Dimensional Scaling (SMDS) Pipeline.

This demo demonstrates the full discovery_pipeline with cross-validation,
CSV export, and visualization generation.

To run this demo:
    python -m smds.pipeline.demo_with_pipeline

Or from the project root:
    cd /path/to/supervised-multidimensional-scaling
    python -m smds.pipeline.demo_with_pipeline
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.continuous_shapes.euclidean import EuclideanShape
from smds.shapes.continuous_shapes.log_linear import LogLinearShape
from smds.shapes.continuous_shapes.semicircular import SemicircularShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.discrete_shapes.cluster import ClusterShape


def generate_circular_data(
    n_samples: int = 100, n_features: int = 30, noise_level: float = 0.1
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    y = (angles / (2 * np.pi)).astype(np.float64)

    embedding_2d = np.column_stack([np.cos(angles), np.sin(angles)])
    random_projection = np.random.randn(2, n_features)
    random_projection /= np.linalg.norm(random_projection, axis=0)

    X = (embedding_2d @ random_projection).astype(np.float64)
    X += np.random.randn(n_samples, n_features) * noise_level

    return X, y


def generate_cluster_data(
    n_samples: int = 100, n_features: int = 30, n_clusters: int = 4, noise_level: float = 0.5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    samples_per_cluster = n_samples // n_clusters
    y = np.repeat(np.arange(n_clusters), samples_per_cluster).astype(np.float64)

    cluster_centers = np.random.randn(n_clusters, n_features) * 10

    X = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(n_clusters):
        start_idx = i * samples_per_cluster
        end_idx = start_idx + samples_per_cluster
        X[start_idx:end_idx] = cluster_centers[i] + np.random.randn(samples_per_cluster, n_features) * noise_level

    return X, y / (n_clusters - 1)


def generate_linear_data(
    n_samples: int = 100, n_features: int = 30, noise_level: float = 0.1
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.linspace(0, 1, n_samples).astype(np.float64)

    direction = np.random.randn(n_features)
    direction /= np.linalg.norm(direction)

    X = np.outer(y, direction).astype(np.float64)
    X += np.random.randn(n_samples, n_features) * noise_level

    return X, y


def main() -> None:
    print("\n" + "=" * 80)
    print("SUPERVISED MULTI-DIMENSIONAL SCALING (SMDS) - PIPELINE DEMO")
    print("=" * 80)
    print("\nThis demo uses the full discovery_pipeline with cross-validation,")
    print("CSV export, and visualization generation.")

    np.random.seed(42)

    all_shapes = [
        CircularShape(),
        ClusterShape(),
        EuclideanShape(),
        LogLinearShape(),
        SemicircularShape(),
        ChainShape(),
    ]

    print("\n\n" + "=" * 80)
    print("DEMO 1: Circular Manifold (with full pipeline)")
    print("=" * 80)
    X_circular, y_circular = generate_circular_data(n_samples=80, noise_level=0.1)

    print("\nRunning discovery pipeline...")
    print("- Cross-validation with 2 folds")
    print("- Saving results to CSV")
    print("- Generating visualization")

    results_df, save_path = discover_manifolds(
        X_circular,
        y_circular,
        shapes=all_shapes,
        n_folds=2,
        n_jobs=1,
        save_results=True,
        experiment_name="pipeline_demo_circular",
        create_visualization=True,
        clear_cache=False,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\nTop 5 Shapes (sorted by mean test score):\n")

    top_5 = results_df.head(5)
    for idx, row in top_5.iterrows():
        marker = "[WINNER]" if idx == 0 else "       "
        print(
            "{0} {1}. {2:<20} Score: {3:.4f} +/- {4:.4f}".format(
                marker, idx + 1, row["shape"], row["mean_test_score"], row["std_test_score"]
            )
        )

    if save_path:
        print("\nFiles saved:")
        print("  CSV:          {0}".format(save_path))
        print("  Visualization: {0}".format(save_path.replace(".csv", "_dashboard.png")))

    print("\n\n" + "=" * 80)
    print("DEMO 2: Linear Manifold (with full pipeline)")
    print("=" * 80)
    X_linear, y_linear = generate_linear_data(n_samples=80, noise_level=0.1)

    print("\nRunning discovery pipeline...")

    results_df2, save_path2 = discover_manifolds(
        X_linear,
        y_linear,
        shapes=all_shapes,
        n_folds=2,
        n_jobs=1,
        save_results=True,
        experiment_name="pipeline_demo_linear",
        create_visualization=True,
        clear_cache=False,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("\nTop 5 Shapes (sorted by mean test score):\n")

    top_5_2 = results_df2.head(5)
    for idx, row in top_5_2.iterrows():
        marker = "[WINNER]" if idx == 0 else "       "
        print(
            "{0} {1}. {2:<20} Score: {3:.4f} +/- {4:.4f}".format(
                marker, idx + 1, row["shape"], row["mean_test_score"], row["std_test_score"]
            )
        )

    if save_path2:
        print("\nFiles saved:")
        print("  CSV:           {0}".format(save_path2))
        print("  Visualization: {0}".format(save_path2.replace(".csv", "_dashboard.png")))

    print("\n\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey differences from simple demo:")
    print("  - Uses discover_manifolds() pipeline with cross-validation")
    print("  - Generates CSV files with detailed results")
    print("  - Creates visualization dashboards (PNG files)")
    print("  - Includes standard deviation from CV folds")
    print("  - Supports caching for faster re-runs")
    print("\nAll results saved to: smds/pipeline/saved_results/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
