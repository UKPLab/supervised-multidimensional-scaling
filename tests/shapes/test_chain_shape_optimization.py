import time
import numpy as np
import pytest
from smds import SupervisedMDS
from smds.shapes.discrete_shapes import ChainShape


def generate_large_synthetic_data(n_samples: int, n_features: int = 768, seed: int = 42):
    """
    Generates a large synthetic dataset representing a latent cycle/chain
    projected into high-dimensional space with noise.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate Latent Structure (2D Circle)
    # This works as a proxy for a Chain structure for testing purposes
    y_labels = np.arange(n_samples).astype(float)
    angles = 2 * np.pi * y_labels / n_samples
    X_latent = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # 2. Project to High Dimensions (e.g., 768 for LLMs)
    projection_matrix = rng.standard_normal((2, n_features))
    X_high = X_latent @ projection_matrix

    # 3. Add Noise
    X_high += rng.normal(scale=0.1, size=X_high.shape)

    return X_high, y_labels


@pytest.mark.benchmark
@pytest.mark.parametrize("n_samples", [1000, 5000])
def test_gpu_acceleration_chain(n_samples):
    """
    Compares Standard Chain (CPU) vs GPU-Accelerated Chain.
    """
    n_features = 768
    X, y = generate_large_synthetic_data(n_samples=n_samples, n_features=n_features)

    print(f"\n\n--- BENCHMARK: N={n_samples}, D={n_features} ---")

    # 1. CPU (Old Way)
    start_time = time.time()
    mds_cpu = SupervisedMDS(n_components=2, manifold=ChainShape(threshold=n_samples * 0.1), gpu_accel=False)
    mds_cpu.fit(X, y)
    cpu_duration = time.time() - start_time
    print(f"CPU Solver (SciPy):  {cpu_duration:.4f} seconds")

    # 2. GPU (New Way)
    start_time = time.time()
    mds_gpu = SupervisedMDS(n_components=2, manifold=ChainShape(threshold=n_samples * 0.1), gpu_accel=True)
    mds_gpu.fit(X, y)
    gpu_duration = time.time() - start_time
    print(f"GPU Solver (Torch):  {gpu_duration:.4f} seconds")

    #factor = cpu_duration / gpu_duration if gpu_duration > 0 else 0
    #print(f"Speedup Factor: {factor:.1f}x")


if __name__ == "__main__":
    # Allows running directly with python test_chain_shape_optimization.py
    # to see output immediately without pytest capturing stdout
    test_gpu_acceleration_chain(1000)

# Todo: this cant be a real test, since the results are hardware dependent. how to make this an optional/manual test?
