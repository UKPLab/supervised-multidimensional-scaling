<p  align="center">
  <img src='logo.png' width='200'>
</p>

# Supervised Multi-Dimensional Scaling
[![Arxiv](https://img.shields.io/badge/Arxiv-2510.01025-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.01025)
[![License](https://img.shields.io/github/license/UKPLab/supervised-multidimensional-scaling)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/supervised-multidimensional-scaling/actions/workflows/main.yml)

This is a stand-alone implementation of Supervised Multi-Dimensional Scaling (SMDS) from the paper "Shape Happens: Automatic Feature Manifold Discovery in LLMs". It contains a plug-and-play class written with the familiar [scikit-learn](https://scikit-learn.org) interface. SMDS supports several template shapes to discover manifolds of various shape.

Contact person: [Federico Tiblias](mailto:federico.tiblias@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue or if you have further questions.


## Getting Started (TODO)

## Usage

The `SupervisedMDS` class provides a scikit-learn style interface that is straightforward to use, along with utilities such as an inverse function and saving/loading a trained model.  

### Fit & Transform

You can instantiate the model, fit it to data `(X, y)`, and transform your input into a low-dimensional embedding:

```python
import numpy as np
from supervised_mds import SupervisedMDS

# Example data
X = np.random.randn(100, 20)   # 100 samples, 20 features
y = np.random.randint(0, 5, size=100)  # Discrete labels (clusters)

# Instantiate and fit
smds = SupervisedMDS(n_components=2, manifold="cluster", alpha=0.1)
smds.fit(X, y)

# Transform to low-dimensional space
X_proj = smds.transform(X)
print(X_proj.shape)  # (100, 2)
```

### Extra functionalities

Once fitted, you can use the learned transformation for inverse projections and to assess how well the embedding matches the target geometry:

```python
# Inverse transform: approximate reconstruction of original features
X_reconstructed = smds.inverse_transform(X_proj)
print(X_reconstructed.shape)  # (100, 20)

# Scoring: measure alignment between transformed distances and ideal distances
score = smds.score(X, y)
print(f"Model score: {score:.3f}")
```

### Saving & loading

Models can be persisted to disk, including the learned transformation, and reloaded later for reuse:

```python
# Save to file
smds.save("smds_model.pkl")

# Load from file
loaded_smds = SupervisedMDS.load("smds_model.pkl")
```

### Optimization & GPU Support

For manifolds with undefined distances (like `ChainShape`), this library defaults to a generic SciPy solver. For large datasets, this can be slow. 

We provide an optional accelerated solver using `PyTorch`, which is faster on CPU and significantly faster on GPU.

### Enabling the Accelerator
1.  Install `PyTorch` (see below).
2.  Initialize the model with `use_gpu=True`:
    ```python
    mds = SupervisedMDS(..., manifold=ChainShape(...), use_gpu=True)
    ```

### Installation Guide

Since hardware requirements vary, `PyTorch` is **not** installed by default.

On **<ins>Windows/Linux</ins>**:
 - **CPU Acceleration [No GPU]**
   -  Running the standard install will enable the optimized Adam solver **on your CPU**. 
      This is faster than the default, even **without a GPU**.
      ```bash
      uv pip install torch
       ```
   


 - **GPU Acceleration [NVIDIA CUDA]** 
    - To enable GPU acceleration, you need the specific CUDA-enabled version of PyTorch. Find the correct Index URL for 
   your system on the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page (under the "Compute Platform" selector).
        ```bash
        uv pip install torch --index-url https://download.pytorch.org/whl/<cuda_version>
        ```
   
    - If you are using Conda, your GPU may be detected automatically when you install PyTorch with:
          ```
          conda install pytorch -c pytorch -c nvidia
          ```

On **<ins>MacOS</ins>** (Apple Silicon M1/M2/M3):
- The standard installation automatically enables GPU acceleration via Apple Metal (MPS).
    ```bash
    uv pip install torch
     ```




## Coming Soon...

- [Feature]: A comprehensive test suite
- [Feature]: Advanced support for hypothesis manifolds
- [Feature]: A manifold discovery utility
- [Docs]: Setup instructions.
- [Docs]: A documentation website with class descriptions and examples.



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
