# How to use SMDS by Shape Wizards?!

This guide provides a complete walkthrough of using the Supervised Multi-Dimensional Scaling (SMDS) library, from basic fitting to automatic manifold discovery and statistical validation.

## 1. Installation

Install `smds` using your preferred package manager.

**Using uv (Recommended):**
```shell
uv add smds
```

**Using pip:**
```bash
pip install smds
```

## 2. Basic Usage: The SupervisedMDS Class

The core of the library is the `SupervisedMDS` class, which offers a scikit-learn compatible interface. Use this when you know the specific manifold shape you want to target.

### Fit and Transform

You can instantiate the model, fit it to your data `(X, y)`, and project it into a low-dimensional space.

```python
import numpy as np
from smds import SupervisedMDS

# 1. Prepare your data
# X: Input features (e.g., embeddings from an LLM)
# y: Labels or coordinates (e.g., class labels, time steps, circular angles)
X = np.random.randn(100, 20)
y = np.random.randint(0, 5, size=100)

# 2. Instantiate the model
# stage_1: Strategy to initialize the distance matrix ('computed' is usually default)
# manifold: The target shape (e.g., 'cluster', 'circular', 'chain', 'hierarchical')
model = SupervisedMDS(stage_1="computed", manifold="cluster", alpha=0.1)

# 3. Fit to data
model.fit(X, y)

# 4. Transform (Project) to low-dimensional space
X_projected = model.transform(X)

print(f"Projected shape: {X_projected.shape}")  # (100, 2)
```

### Parameters
- **`manifold`**: The name of the target shape (e.g., `"cluster"`, `"circular"`). See [Available Shapes](shapes.md) for a full list.
- **`stage_1`**: Strategy for the first stage. `"computed"` is standard. `"user_provided"` requires passing a pre-computed distance matrix.
- **`alpha`**: Regularization parameter (if applicable).

## 3. Automatic Manifold Discovery

If you don't know the underlying geometry of your data, you can use the **Discovery Pipeline**. This tool tests multiple manifold hypotheses (shapes) and ranks them by how well they fit your data.

```python
from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.pipeline import open_dashboard

# Run the discovery pipeline
# This will evaluate default shapes like Cluster, Circular, Hierarchical, etc.
df_results, save_path = discover_manifolds(
    X, 
    y, 
    smds_components=2,           # Target dimensionality
    n_folds=5,                   # 5-Fold Cross-Validation for robust scoring
    experiment_name="My_Discovery_Exp",
    n_jobs=-1                    # Use all CPU cores
)

# View the best matching shape
print(f"Best matching shape: {df_results.iloc[0]['shape']}")
print(df_results.head())
```

## Next Steps

- Check out the [Available Shapes](shapes.md) to see what geometries you can model.
- Learn how to define your own [Custom Shape](examples/custom_shape.md).
