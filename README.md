<p  align="center">
  <img src='logo.png' width='200'>
</p>

# Supervised Multi-Dimensional Scaling

[![Arxiv](https://img.shields.io/badge/Arxiv-2510.01025-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.01025)
[![License](https://img.shields.io/github/license/UKPLab/supervised-multidimensional-scaling)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml)

This is a stand-alone implementation of Supervised Multi-Dimensional Scaling (SMDS) from the paper "Shape Happens: Automatic Feature Manifold Discovery in LLMs". It contains a plug-and-play class written with the familiar [scikit-learn](https://scikit-learn.org) interface. SMDS supports several template shapes to discover manifolds of various forms.

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

The library provided a high level pipeline to automatically discover the intrinsic manifold of your data. The `discover_manifolds` utility evaluates a set of hypothesis shapes (for example: clusters, circles, hierarchies) and ranks them based on how well they explain the data structure using cross validation.

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

> [!NOTE]
> The `dev` dependency group includes heavy libraries such as `torch` and `transformers`.


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
