from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.continuous_shapes.euclidean import EuclideanShape
from smds.shapes.continuous_shapes.log_linear import LogLinearShape
from smds.shapes.discrete_shapes.cluster import ClusterShape

# --- 1. Data Generator (Simulating LLM Neurons) ---


def generate_synthetic_llm_data(
    n_samples: int = 300, n_dim: int = 50, topology: str = "circular", noise: float = 0.08
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates artificial "Hidden States" (X) and Labels (y).
    The data lies on a specific manifold embedded in a high-dimensional space.
    """
    np.random.seed(42)

    # y are our labels (e.g., time points), normalized to [0, 1]
    y = np.linspace(0, 1, n_samples)

    if topology == "circular":
        # Simulates "Months": Jan (0.0) is close to Dec (1.0)
        # 2D Embedding: Circle
        angle = 2 * np.pi * y
        manifold_2d = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    elif topology == "linear":
        # Simulates "Years": 1900 is far away from 2000
        # 2D Embedding: Line (with slight curvature, often seen in real data)
        manifold_2d = np.stack([y, 0.5 * y**2], axis=1)

    elif topology == "cluster":
        # Simulates "Categories": Spring, Summer, Autumn, Winter
        n_clusters = 4
        y_int = (y * n_clusters).astype(int)  # 0, 1, 2, 3
        manifold_2d = np.zeros((n_samples, 2))
        centers = [(0, 0), (1, 1), (1, 0), (0, 1)]
        for i in range(n_samples):
            c = centers[y_int[i]]
            manifold_2d[i] = c + np.random.normal(0, 0.05, 2)
    else:
        raise ValueError("Unknown topology")

    # Projection into the "High-Dimensional Space" (e.g., 4096 dim in Llama, here 128)
    # We rotate the 2D structure randomly into the 128D space
    projection_matrix = np.random.randn(2, n_dim)
    X = manifold_2d @ projection_matrix

    # Add noise (LLM representations are never perfect)
    X += np.random.normal(0, noise, X.shape)

    return X, y


# --- 2. Run Experiments ---


def run_demo() -> None:
    print("🚀 Starting SMDS Manifold Discovery Demo...\n")

    # List of shapes to test (our hypotheses)
    # We intentionally include "wrong" shapes to see if the ranking works
    shapes_to_test = [
        CircularShape(),  # Expected winner for Circle
        EuclideanShape(),  # Expected winner for Line (Linear)
        LogLinearShape(),  # Tests for logarithmic scaling
        ClusterShape(),  # Tests for clusters
    ]

    # --- EXPERIMENT 1: Circular Data (e.g., Months) ---
    print("-" * 60)
    print("🧪 Experiment 1: Simulating 'Month Data' (Circular)")
    X_circ, y_circ = generate_synthetic_llm_data(topology="circular")

    df_circ, path_circ = discover_manifolds(
        X=X_circ,
        y=y_circ,
        shapes=shapes_to_test,
        n_folds=5,
        save_results=True,
        experiment_name="Demo_Months_Circular",
        create_visualization=True,  # This triggers your new plotting code
    )

    print("\n🏆 Top 3 Results (Circular Experiment):")
    print(df_circ[["shape", "mean_test_score", "std_test_score"]].head(3))
    if path_circ:
        print(f"📊 Dashboard saved to: {path_circ.replace('.csv', '_dashboard.png')}")

    # --- EXPERIMENT 2: Linear Data (e.g., Years) ---
    print("\n" + "-" * 60)
    print("🧪 Experiment 2: Simulating 'Year Data' (Linear)")
    X_lin, y_lin = generate_synthetic_llm_data(topology="linear")

    df_lin, path_lin = discover_manifolds(
        X=X_lin,
        y=y_lin,
        shapes=shapes_to_test,
        n_folds=5,
        save_results=True,
        experiment_name="Demo_Years_Linear",
        create_visualization=True,
    )

    print("\n🏆 Top 3 Results (Linear Experiment):")
    print(df_lin[["shape", "mean_test_score", "std_test_score"]].head(3))
    if path_lin:
        print(f"📊 Dashboard saved to: {path_lin.replace('.csv', '_dashboard.png')}")


if __name__ == "__main__":
    run_demo()
