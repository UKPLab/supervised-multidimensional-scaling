"""
Entry point for running the SMDS Discovery Pipeline demo.
"""

import numpy as np
from numpy.typing import NDArray

from smds.pipeline import open_dashboard
from smds.pipeline.discovery_pipeline import discover_manifolds

# Configuration
AUTO_OPEN_DASHBOARD = True


def run_demo() -> None:
    # Generate Demo Data (100 samples, 10 dimensions, circular topology)
    # Using linspace ensures it has a logical continuous structure
    X: NDArray[np.float64] = np.random.randn(99, 10)
    y: NDArray[np.float64] = np.linspace(0, 1, 99)

    print("Starting Manifold Discovery...")

    # Execute Pipeline
    df, saved_file_path = discover_manifolds(
        X,
        y,
        experiment_name="Discovery_Demo",
        save_results=True,
        create_visualization=True,
        clear_cache=True
    )

    print(f"\nAnalysis Complete. Winner: {df.iloc[0]['shape']}")

    # Launch Dashboard
    if AUTO_OPEN_DASHBOARD:
        open_dashboard.main(saved_file_path)


if __name__ == "__main__":
    run_demo()
