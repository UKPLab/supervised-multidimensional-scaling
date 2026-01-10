"""
Manual / Optional GPU Benchmark Test

This module contains a manual performance benchmark for testing GPU
acceleration in SupervisedMDS with a ChainShape manifold.

IMPORTANT:
- This is **not** a deterministic unit test.
- Results are **hardware-dependent** (GPU model, drivers, CUDA version, etc.).
- It is **not intended to run as part of the standard CI test pipeline**.

Purpose:
- Compare relative runtime behavior of CPU vs GPU solvers on large,
  high-dimensional synthetic data.
"""

import time

import numpy as np
from numpy.typing import NDArray

from smds import SupervisedMDS
from smds.shapes.discrete_shapes import ChainShape


def generate_large_synthetic_data(n_samples: int, n_features: int = 768, seed: int = 42) -> tuple[NDArray[np.float64],
NDArray[np.float64]]:
    """
    Generates a large synthetic dataset representing a latent cycle/chain
    projected into high-dimensional space with noise.
    """
    rng = np.random.default_rng(seed)

    # Generate Latent 2D Circle
    # This works as a proxy for a Chain structure for testing purposes
    y_labels = np.arange(n_samples).astype(float)
    angles = 2 * np.pi * y_labels / n_samples
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Project to High Dimensions
    projection_matrix = rng.standard_normal((2, n_features))
    X_high = X_latent @ projection_matrix

    # Add Noise
    X_high += rng.normal(scale=0.1, size=X_high.shape)

    return X_high, y_labels


def run_gpu_acceleration_chain(n_samples: int, compare_to_cpu: bool) -> None:
    """
    Executes a performance benchmark for the GPU-accelerated ChainShape solver.

    This test generates a large synthetic dataset (simulating LLM hidden states with
    768 features) and fits a SupervisedMDS model using the PyTorch-based optimizer.

    Context:
        The ChainShape produces a sparse distance matrix (containing undefined `-1.0` values),
        which forces SupervisedMDS to bypass the fast spectral solution (Classical MDS)
        and use iterative gradient descent. This test validates that the PyTorch path
        executes successfully and prints the wall-clock runtime.

    Args:
        n_samples (int): The number of samples to generate. Higher values (e.g., 5000)
                         effectively demonstrate the scaling benefits of the GPU solver
                         compared to the CPU baseline.

    Note:
        The CPU comparison (SciPy solver) is currently commented out to allow for
        quick sanity checks of the GPU path without waiting for the slow CPU convergence.
    """
    n_features = 768
    X, y = generate_large_synthetic_data(n_samples=n_samples, n_features=n_features)

    print(f"\nBENCHMARK: N={n_samples}, D={n_features}")

    # GPU accelerated Solver (Torch)
    print("--- GPU Solver ---")
    start_time = time.time()
    mds_gpu = SupervisedMDS(n_components=2, manifold=ChainShape(threshold=n_samples * 0.1), gpu_accel=True)
    mds_gpu.fit(X, y)
    gpu_duration = time.time() - start_time
    print(f"GPU Solver (Torch):  {gpu_duration:.4f} seconds\n")

    # Standard SciPy Solver (Optional Comparison)
    if compare_to_cpu:
        print("--- CPU Solver ---")
        start_time = time.time()
        mds_cpu = SupervisedMDS(n_components=2, manifold=ChainShape(threshold=n_samples * 0.1), gpu_accel=False)
        mds_cpu.fit(X, y)
        cpu_duration = time.time() - start_time
        print(f"Standard CPU Solver: {cpu_duration:.4f} seconds")

        factor = cpu_duration / gpu_duration if gpu_duration > 0 else 0
        print(f"Speedup Factor: {factor:.1f}x")


if __name__ == "__main__":
    run_gpu_acceleration_chain(500, compare_to_cpu=True)
    run_gpu_acceleration_chain(5000, compare_to_cpu=False)
