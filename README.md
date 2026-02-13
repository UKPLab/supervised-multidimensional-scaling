<p  align="center">
  <img src='logo.png' width='200'>
</p>

# Supervised Multi-Dimensional Scaling

[![Arxiv](https://img.shields.io/badge/Arxiv-2510.01025-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.01025)
[![License](https://img.shields.io/github/license/UKPLab/supervised-multidimensional-scaling)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml)

This is a stand-alone implementation of Supervised Multi-Dimensional Scaling (SMDS) from the paper "Shape Happens:
Automatic Feature Manifold Discovery in LLMs". It contains a plug-and-play class written with the
familiar [scikit-learn](https://scikit-learn.org) interface. SMDS supports several template shapes to discover manifolds
of various shape.

Contact person: [Federico Tiblias](mailto:federico.tiblias@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to report an issue if you have further questions or spot a bug.

## Getting started

With uv (recommended):
```shell
uv add smds
```

With pip:
```bash
pip install smds
```

## Usage

The `SupervisedMDS` class provides a scikit-learn style interface that is straightforward to use. Unlike standard MDS, it requires selecting a stage-1 strategy and target `manifold` by name (for example: `"cluster"`, `"circular"`).

### Fit & Transform

You can instantiate the model, fit it to data `(X, y)`, and transform your input into a low-dimensional embedding:

```python
import numpy as np
from smds import SupervisedMDS

# Example data
X = np.random.randn(100, 20)  # 100 samples, 20 features
y = np.random.randint(0, 5, size=100)  # Discrete labels (clusters)

# Instantiate and fit
# stage_1: "computed" (default) or "user_provided"
# manifold: one of the built-in shape names, e.g. "cluster", "circular", "log_linear"
smds = SupervisedMDS(stage_1="computed", manifold="cluster", alpha=0.1)
smds.fit(X, y)

# Transform to low-dimensional space
X_proj = smds.transform(X)
print(X_proj.shape)  # (100, 2)
```

If you set `stage_1="user_provided"`, `manifold` is ignored and a warning is raised.

### Manifold Discovery

Once fitted, you can use the learned transformation for inverse projections and to assess how well the embedding matches
the target geometry:

```python
from smds.pipeline.discovery_pipeline import discover_manifolds
from smds.pipeline import open_dashboard

# Run discovery pipeline
# Evaluates default shapes (Cluster, Circular, Hierarchical, etc.)
# Returns a DataFrame sorted by best fit (lowest stress / highest score)
df_results, save_path = discover_manifolds(
    X, 
    y, 
    smds_components=2,           # Target dimensionality
    n_folds=5,                   # Cross-validation folds
    experiment_name="My_Exp",    # Name for saved results
    n_jobs=-1                    # Use all available cores
)

print(f"Best matching shape: {df_results.iloc[0]['shape']}")
print(df_results.head())

# Launch the interactive Streamlit dashboard to explore results and plots
open_dashboard.main(save_path)
```

The discovery pipeline handles:
- **Hypothesis Testing**: Iterates through a default or custom list of manifold shapes.
- **Cross-Validation**: Uses k-fold CV to ensure robust scoring.
- **Caching**: Caches intermediate results to resume interrupted experiments.
- **Visualization**: Generates interactive plots for the dashboard.

## Development

This seciton is especially usefull if you consider contributing to the library!

### Documentation

To build and serve the documentation locally:

```bash
mkdocs serve
```

### Statistical Validation

Standard cross-validation provides a mean score, but it does not tell you if one manifold is *statistically* better than another. SMDS includes a robust **Statistical Testing (ST)** wrapper that runs repeated experiments to perform a **Friedman Rank Sum Test** and **Nemenyi Post-Hoc Analysis**.

#### Running a Statistical Test

Instead of `smds/pipeline/run_pipeline`, use the `smds/pipeline/run_statistical_test.py` wrapper:

```python
from smds.pipeline.statistical_testing.run_statistical_test import run_statistical_validation

# Runs the pipeline 10 times (10 repeats), each with 5-Fold CV
pivot_dfs, output_path = run_statistical_validation(
    X=my_data, 
    y=my_labels, 
    n_repeats=10,
    n_folds=5,
    experiment_name="my_robust_experiment"
)
```

#### Viewing Results

Open the dashboard to view the **Friedman Statistic**, **P-Value Heatmap**, and **Critical Difference (CD) Diagram**:

```bash
python smds/pipeline/open_dashboard.py
```

## Optimization & GPU Support

For manifolds with undefined distances (e.g. `ChainShape`), SMDS falls back to a generic SciPy solver.  
For large datasets, this can be slow.

SMDS provides an **optional accelerated solver** based on **PyTorch**, which is significantly faster on CPU and can transparently leverage GPUs when available.

---

### Enabling the Accelerator

Install SMDS with the optional `fast` extra:

```bash
  pip install smds[fast]
```

Then enable it in your model:
```python
smds = SupervisedMDS(
    ...,
    manifold=ChainShape(...),
    use_gpu=True,
)
```
If a compatible GPU is available, PyTorch will use it automatically.
Otherwise, the accelerated solver will run on CPU.

>💡 GPU support (CUDA on NVIDIA, MPS on Apple Silicon) depends on your PyTorch installation.
See the official PyTorch documentation for platform-specific setup.



## Coming Soon...

### Testing

Run the test suite using pytest:

```bash
make test
```

## Contributors

<a href="https://github.com/UKPLab/supervised-multidimensional-scaling/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=UKPLab/supervised-multidimensional-scaling" />
</a>

## Cite

Please use the following citation:

```
@misc{tiblias2025shapehappensautomaticfeature,
      title={Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling}, 
      author={Federico Tiblias and Irina Bigoulaeva and Jingcheng Niu and Simone Balloccu and Iryna Gurevych},
      year={2025},
      eprint={2510.01025},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.01025}, 
}
```
