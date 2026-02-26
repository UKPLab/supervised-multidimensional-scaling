import os
import pickle
import warnings
from abc import ABC, abstractmethod
from math import exp
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin, clone  # type: ignore[import-untyped]
from sklearn.utils.multiclass import type_of_target  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

from smds.shapes.continuous_shapes import (
    CircularShape,
    EuclideanShape,
    KleinBottleShape,
    LogLinearShape,
    SemicircularShape,
    SpiralShape,
)
from smds.shapes.discrete_shapes import (
    ChainShape,
    ClusterShape,
    DiscreteCircularShape,
    GraphGeodesicShape,
    HierarchicalShape,
    PolytopeShape,
)
from smds.shapes.spatial_shapes import CylindricalShape, GeodesicShape, SphericalShape
from smds.stress import (
    StressMetrics,
    kl_divergence_stress,
    non_metric_stress,
    normalized_stress,
    scale_normalized_stress,
    shepard_goodness_stress,
)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# TODO: stage 2 - for mapping to the lower space
#   result: X_proj


# smds stage 1 - for the manifold Y_
class SMDSParametrization(TransformerMixin, BaseEstimator, ABC):  # type: ignore[misc]
    @property
    @abstractmethod
    def n_components(self) -> int | None:
        """
        Subclasses must implement this.
        Number of components of the projected manifold.

        Returns
        -------
        n_components : int
            Number of components of the projected manifold.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> "SMDSParametrization":
        """
        Subclasses must implement this.
        It is required for TransformerMixin.fit_transform to work.

        Parameters
        ----------
        X : ndarray
            Input labels or coordinates.
        y : ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : SMDSParametrization
            Fitted transformer.
        """
        pass

    @abstractmethod
    def transform(self, X: NDArray[Any] | None = None) -> NDArray[np.float64]:
        """
        Subclasses must implement this.
        It is required for TransformerMixin.fit_transform to work.

        Parameters
        ----------
        X : ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        Y : ndarray
            The embedding coordinates.
        """
        pass

    @abstractmethod
    def compute_ideal_distances(self, y: NDArray[Any]) -> NDArray[np.float64]:
        """
        Subclasses must implement this.
        Return the pairwise distance matrix for the given labels or coordinates.

        Parameters
        ----------
        y : ndarray
            Input labels or coordinates.

        Returns
        -------
        D : ndarray
            Pairwise distance matrix.
        """
        pass


class ComputedSMDSParametrization(SMDSParametrization):
    """
    Stage-1 parametrization that computes ideal distances using a manifold function.

    Parameters
    ----------
    manifold : Callable
        Function that takes labels and returns a distance matrix.
    n_components : int
        Number of embedding dimensions.
    """

    def __init__(self, manifold: Callable[[NDArray[Any]], NDArray[np.float64]], n_components: int):
        # fixme: set manifold to be BaseShape
        self.manifold = manifold
        self._n_components = n_components

    @property
    def n_components(self) -> int:
        """
        Number of manifold coordinates produced by this stage.
        """
        return self._n_components

    def compute_ideal_distances(self, y: NDArray[Any], threshold: int = 2) -> NDArray[np.float64]:
        """
        Compute ideal pairwise distance matrix from labels.

        Parameters
        ----------
        y : ndarray
            Input labels or coordinates.
        threshold : int, default=2
            Distance threshold parameter.

        Returns
        -------
        D : ndarray
            Pairwise distance matrix.
        """
        if callable(self.manifold):
            D: np.ndarray = self.manifold(y)
        else:
            raise ValueError("Invalid manifold specification.")

        return D

    def _classical_mds(self, D: NDArray[Any]) -> NDArray[Any]:
        """
        Perform classical MDS on distance matrix.

        Parameters
        ----------
        D : ndarray
            Pairwise distance matrix.

        Returns
        -------
        Y : ndarray
            Low-dimensional embedding coordinates.
        """
        # Square distances
        D2 = D**2

        # Double centering
        n = D2.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D2 @ H

        # Eigen-decomposition
        eigvals, eigvecs = eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][: self.n_components]
        eigvecs = eigvecs[:, idx][:, : self.n_components]

        # Embedding computation
        Y: np.ndarray = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        return Y

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> "ComputedSMDSParametrization":
        """
        Fit by computing ideal distances and MDS embedding.

        Parameters
        ----------
        X : ndarray
            Input labels or coordinates.
        y : ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : ComputedSMDSParametrization
            Fitted transformer.
        """
        self.D_ = self.compute_ideal_distances(X)
        self.Y_ = self._classical_mds(self.D_)
        return self

    def transform(self, X: NDArray[Any] | None = None) -> NDArray[np.float64]:
        """
        Return the computed embedding.

        Parameters
        ----------
        X : ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        Y : ndarray
            The embedding coordinates.
        """
        return self.Y_


class UserProvidedSMDSParametrization(SMDSParametrization):
    """
    Stage-1 parametrization using user-provided coordinates or template mapping.

    Parameters
    ----------
    y : ndarray, optional
        Pre-computed embedding coordinates.
    n_components : int, optional
        Number of embedding dimensions (inferred from y if not provided).
    fixed_template : ndarray, optional
        Fixed template coordinates for mapping.
    mapper : Callable, optional
        Function to map labels to template coordinates.
    name : str, optional
        Name for this parametrization.
    """

    def __init__(
        self,
        y: NDArray[Any] | None = None,
        n_components: int | None = None,
        fixed_template: NDArray[np.float64] | None = None,
        mapper: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None = None,
        name: str | None = None,
    ):
        self._n_components = n_components
        self.fixed_template = fixed_template
        self.mapper = mapper
        self.name = name
        self.y = y

        if self.y is not None:
            self.y = np.asarray(self.y)
            if self.y.ndim == 1:
                self.y = self.y.reshape(-1, 1)
            inferred = self.y.shape[-1]
            if self._n_components is None:
                self._n_components = inferred
            elif self._n_components != inferred:
                raise ValueError(
                    f"y must have shape compatible with n_components ({self._n_components}), got {self.y.shape}"
                )

    @property
    def n_components(self) -> int | None:
        """
        Number of manifold coordinates represented by the provided embedding.
        """
        return self._n_components

    def _calc_dist(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        dist: NDArray[np.float64] = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
        return dist

    def compute_ideal_distances(self, y: NDArray[Any] | None = None) -> NDArray[np.float64]:
        """
        Compute pairwise distances from stored or provided coordinates.

        Parameters
        ----------
        y : ndarray, optional
            Coordinates to compute distances from. If None, uses stored Y_.

        Returns
        -------
        D : ndarray
            Pairwise distance matrix.
        """
        if self.y is not None:
            return self._calc_dist(self.Y_)

        if y is None:
            return self._calc_dist(self.Y_)

        y = np.asarray(y)

        if self.fixed_template is not None and self.mapper is not None:
            coords = self.mapper(y.squeeze(), self.fixed_template)
            return self._calc_dist(coords)
        else:
            return self._calc_dist(y)

    def fit(
        self,
        X: NDArray[Any] | None = None,
        y: NDArray[Any] | None = None,
    ) -> "UserProvidedSMDSParametrization":
        """
        Store coordinates and compute distance matrix.

        Parameters
        ----------
        X : ndarray, optional
            Coordinates (used if y is None).
        y : ndarray, optional
            Coordinates (preferred over X).

        Returns
        -------
        self : UserProvidedSMDSParametrization
            Fitted transformer.
        """
        if self.y is not None:
            self.Y_ = self.y
            self.D_ = self.compute_ideal_distances(None)
            return self

        target_data = y if y is not None else X

        if target_data is None:
            raise ValueError("UserProvidedSMDSParametrization requires y in fit(X, y) or constructor.")

        target_data = np.asarray(target_data)

        if self.fixed_template is not None and self.mapper is not None:
            mapped_coords = self.mapper(target_data.squeeze(), self.fixed_template)
            mapped_coords = np.asarray(mapped_coords)
            if mapped_coords.ndim == 1:
                mapped_coords = mapped_coords.reshape(-1, 1)
            self.Y_ = mapped_coords
        else:
            if target_data.ndim == 1:
                target_data = target_data.reshape(-1, 1)
            elif target_data.ndim != 2:
                raise ValueError(f"y must be 1D or 2D. Got shape {target_data.shape}.")

            if self._n_components is None:
                self._n_components = target_data.shape[1]
            elif target_data.shape[1] != self._n_components:
                raise ValueError(
                    f"y must have shape compatible with n_components ({self._n_components}), got {target_data.shape}"
                )
            self.Y_ = target_data

        self.D_ = self.compute_ideal_distances(None)
        return self

    def transform(self, X: NDArray[Any] | None = None) -> NDArray[np.float64]:
        """
        Return the stored embedding.

        Parameters
        ----------
        X : ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        Y : ndarray
            The embedding coordinates.
        """
        return self.Y_


class SupervisedMDS(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    _STAGE_1_OPTIONS = {"computed", "user_provided"}

    def __init__(
        self,
        stage_1: str = "computed",
        manifold: str = "circular",
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
        gpu_accel: bool = False,
    ):
        # todo: add string for stage_1
        # todo: add string for manifold
        # todo: warn if stage_1 is UserProv -> manifold is ignored
        """
        Parameters
        ----------
            stage_1:
                Stage 1 strategy. One of {"computed", "user_provided"}.
            manifold:
                Manifold type used by computed stage_1.
            metric:
                The metric to use for scoring the embedding.
            gpu_accel:
                If True, attempts to use PyTorch (and CUDA/MPS if available)
                to solve sparse/incomplete manifold optimization.
        """
        self.stage_1 = stage_1
        self.manifold = manifold
        self.alpha = alpha
        self.orthonormal = orthonormal
        self.radius = radius
        self.gpu_accel = gpu_accel

    @staticmethod
    def _normalize_stage_1_name(stage_1: str) -> str:
        if not isinstance(stage_1, str):
            raise TypeError(f"stage_1 must be a string or SMDSParametrization instance, got {type(stage_1).__name__}")
        stage_1_name = stage_1.strip().lower()
        if stage_1_name not in SupervisedMDS._STAGE_1_OPTIONS:
            valid = sorted(SupervisedMDS._STAGE_1_OPTIONS)
            raise ValueError(f"Unknown stage_1: {stage_1!r}. Valid options are: {valid}")
        return stage_1_name

    @staticmethod
    def _normalize_manifold_name(manifold: str) -> str:
        if not isinstance(manifold, str):
            raise TypeError(f"manifold must be a string, got {type(manifold).__name__}")
        manifold_name = manifold.strip().lower()
        return manifold_name

    def _build_manifold(self, manifold_name: str) -> tuple[Callable[[NDArray[Any]], NDArray[np.float64]], int]:
        manifold_factories: dict[str, tuple[Callable[[], Callable[[NDArray[Any]], NDArray[np.float64]]], int]] = {
            "chain": (lambda: ChainShape(), 2),
            "cluster": (lambda: ClusterShape(), 2),
            "discrete_circular": (lambda: DiscreteCircularShape(), 2),
            "hierarchical": (lambda: HierarchicalShape(level_distances=np.array([100.0, 10.0, 1.0])), 2),
            "circular": (lambda: CircularShape(), 2),
            "cylindrical": (lambda: CylindricalShape(radius=self.radius), 3),
            "spherical": (lambda: SphericalShape(radius=self.radius), 3),
            "geodesic": (lambda: GeodesicShape(radius=self.radius), 3),
            "spiral": (lambda: SpiralShape(), 2),
            "log_linear": (lambda: LogLinearShape(), 1),
            "euclidean": (lambda: EuclideanShape(), 1),
            "semicircular": (lambda: SemicircularShape(), 2),
            "klein_bottle": (lambda: KleinBottleShape(), 4),
            "torus": (lambda: KleinBottleShape(), 3),
            "graph_geodesic": (lambda: GraphGeodesicShape(), 3),
            "polytope": (lambda: PolytopeShape(), 3),
        }
        if manifold_name not in manifold_factories:
            valid = sorted(manifold_factories)
            raise ValueError(f"Unknown manifold: {manifold_name!r}. Valid options are: {valid}")
        factory, n_components = manifold_factories[manifold_name]
        return factory(), n_components

    def _build_stage_1(self, stage_1_name: str, manifold_name: str) -> SMDSParametrization:
        if stage_1_name == "computed":
            manifold_obj, n_components = self._build_manifold(manifold_name)
            return ComputedSMDSParametrization(manifold=manifold_obj, n_components=n_components)

        warnings.warn("stage_1='user_provided': manifold value is ignored.", UserWarning, stacklevel=2)
        return UserProvidedSMDSParametrization()

    def _resolve_stage_1(self) -> SMDSParametrization:
        if isinstance(self.stage_1, SMDSParametrization):
            return self.stage_1
        normalized_stage_1 = self._normalize_stage_1_name(self.stage_1)
        normalized_manifold = self._normalize_manifold_name(self.manifold)
        return self._build_stage_1(normalized_stage_1, normalized_manifold)

    def _validate_and_convert_metric(self, metric: str | StressMetrics) -> StressMetrics:
        """
        Validate and convert the metric to a StressMetrics enum.
        """
        if isinstance(metric, StressMetrics):
            return metric
        valid_metrics = {m.value for m in StressMetrics}
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}. Valid options are: {sorted(valid_metrics)}")
        return StressMetrics(metric)

    def _masked_loss(self, W_flat: np.ndarray, X: np.ndarray, D: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute the loss only on the defined distances (where mask is True).
        """
        n_components = self.stage_1_fitted_.n_components
        if n_components is None:
            raise ValueError("stage_1_fitted_.n_components is not set.")
        W = W_flat.reshape((n_components, X.shape[1]))
        X_proj = (W @ X.T).T
        D_pred = np.linalg.norm(X_proj[:, None, :] - X_proj[None, :, :], axis=-1)
        loss = (D_pred - D)[mask]
        result: float = float(np.sum(loss**2))
        return result

    def _validate_data(
        self, X: np.ndarray, y: np.ndarray, reset: bool = True, stage_1_model: SMDSParametrization | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and process X and y based on the manifold's expected y dimensionality.
        """
        model = self.stage_1_fitted_ if hasattr(self, "stage_1_fitted_") else stage_1_model or self._resolve_stage_1()

        if isinstance(model, UserProvidedSMDSParametrization):
            X = check_array(X)
            y = np.asarray(y)
            if y.ndim not in (1, 2):
                raise ValueError(f"Input 'y' must be 1-dimensional or 2-dimensional, but got shape {y.shape}.")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
                )
            return X, y
        else:
            expected_y_ndim = getattr(model.manifold, "y_ndim", 1)

        if expected_y_ndim == 1:
            X, y = validate_data(self, X, y, reset=reset)
            type_of_target(y, raise_unknown=True)
            y = np.asarray(y).squeeze()
            if y.ndim == 0:
                y = y.reshape(1)
        else:
            if y.ndim != expected_y_ndim:
                raise ValueError(
                    f"Input 'y' must be {expected_y_ndim}-dimensional, "
                    f"but got shape {y.shape} with {y.ndim} dimensions."
                )
            if hasattr(self.stage_1, "validate_y"):
                self.stage_1.validate_y(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
                )

        return X, y

    def _fit_pytorch(self, X: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Specialized solver using PyTorch (AutoDiff + GPU acceleration).
        Much faster for large N than scipy.optimize.minimize.
        """
        # Device Selection & Debugging
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
            print(f"Info: PyTorch solver active. Using GPU: {device_name}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print("Warning: gpu_accel=True was requested, but PyTorch cannot find a CUDA or MPS device.")
            print("         - torch.cuda.is_available():", torch.cuda.is_available())
            print("         - torch.backends.mps.is_available():", torch.backends.mps.is_available())
            print("         See README for CUDA installation instructions.")
            print("         Falling back to PyTorch CPU implementation.")

        # Data Transfer
        # Convert inputs to float32 for speed
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        D_t = torch.tensor(D, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)

        # Parameter Initialization
        n_features = X.shape[1]
        n_components = self.stage_1_fitted_.n_components
        if n_components is None:
            raise ValueError("Stage 1 n_components is not set")

        W_t = torch.nn.Parameter(torch.randn(n_components, n_features, device=device, dtype=torch.float32) * 0.01)

        # Optimization Setup
        optimizer = torch.optim.Adam([W_t], lr=0.01)

        # Convergence settings
        max_epochs = 2000
        tol = 1e-4
        prev_loss = float("inf")

        # Training Loop
        for epoch in range(max_epochs):
            optimizer.zero_grad()

            # Forward: Project X -> X_proj
            # Shape: (N, n_components)
            X_proj = torch.matmul(X_t, W_t.T)

            # Compute pairwise Euclidean distances (highly optimized on GPU)
            D_pred = torch.cdist(X_proj, X_proj, p=2)

            # Masked Loss (MSE on defined distances only)
            diff = D_pred - D_t

            loss = torch.mean(torch.square(diff[mask_t]))

            # Backward
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            # Early Stopping Check (every 50 epochs)
            if epoch % 50 == 0:
                curr_loss = loss.item()
                if abs(prev_loss - curr_loss) < tol:
                    break
                prev_loss = curr_loss

        return W_t.detach().cpu().numpy()

    def _fit_scipy(
        self,
        X: NDArray[np.float64],
        D: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """
        Solver using SciPy (L-BFGS-B) for CPU-based optimization.
        Used when distances are incomplete (negative) and GPU accel is off/unavailable.
        """
        rng = np.random.default_rng(42)
        n_components = self.stage_1_fitted_.n_components
        if n_components is None:
            raise ValueError("Stage 1 n_components is not set")

        W0 = rng.normal(scale=0.01, size=(n_components, X.shape[1]))
        result = minimize(self._masked_loss, W0.ravel(), args=(X, D, mask), method="L-BFGS-B")
        x = cast(np.ndarray, result.x)
        return x.reshape((n_components, X.shape[1]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedMDS":
        """
        Fit the linear transformation W to match distances induced by labels y.
        Uses classical MDS + closed-form when all distances are defined,
        and switches to optimization if some distances are undefined (negative).

        Parameters
        ----------
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2)
                The labels or coordinates defining the ideal distances.

        Returns
        -------
            self: returns an instance of self.
        """
        stage_1_model = self._resolve_stage_1()
        X, y = self._validate_data(X, y, stage_1_model=stage_1_model)

        if X.shape[0] == 1:
            raise ValueError("Found array with n_samples=1. SupervisedMDS requires at least 2 samples.")
        if self.orthonormal and self.alpha != 0:
            print("Warning: orthonormal=True and alpha!=0. alpha will be ignored.")

        X = np.asarray(X)
        y = np.asarray(y).squeeze()

        if (
            isinstance(stage_1_model, UserProvidedSMDSParametrization)
            and stage_1_model.y is None
            and stage_1_model.fixed_template is None
        ):
            y_arr = np.asarray(y)
            n_comp = y_arr.shape[1] if y_arr.ndim > 1 else 1
            if stage_1_model.n_components != n_comp:
                stage_1_model = UserProvidedSMDSParametrization(y=None, n_components=n_comp)

        self.stage_1_fitted_: SMDSParametrization = clone(stage_1_model)
        self.stage_1_fitted_.fit(y)

        self.Y_ = self.stage_1_fitted_.Y_

        D = self.stage_1_fitted_.D_

        if np.any(D < 0):
            # Inform if any distances are negative
            print("Info: Distance matrix is incomplete.")
            mask = D >= 0
            if self.gpu_accel:
                # PyTorch GPU Solver
                if _TORCH_AVAILABLE:
                    print("Info: Using PyTorch solver for sparse manifold.")
                    self.W_ = self._fit_pytorch(X, D, mask)
                else:
                    print(
                        "ImportError: You requested accelerated optimization (gpu_accel=True), "
                        "but PyTorch is not installed.\n\n"
                        "Please install PyTorch to use this feature:\n"
                        "  - Standard (Mac/CPU): uv pip install torch\n"
                        "  - NVIDIA GPU: See README for CUDA installation instructions."
                    )
                    print("\nFalling back to SciPy CPU solver.")
                    self.W_ = self._fit_scipy(X, D, mask)
            else:
                # SciPy CPU Solver (Fallback)
                print(
                    "Warning: Using the SciPy CPU solver for incomplete distance matricies may take a long time. "
                    "Consider setting gpu_accel=True"
                )
                self.W_ = self._fit_scipy(X, D, mask)

        else:
            # Complete Distance Matrix Case (Use Classical MDS + Closed Form)
            Y = self.Y_

            # Using logic from MAIN branch (handles centering/orthonormal better)
            self._X_mean = X.mean(axis=0)  # Centering
            self._Y_mean = Y.mean(axis=0)  # Centering Y
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)

            if self.orthonormal:
                # Orthogonal Procrustes
                M = Y_centered.T @ X_centered
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                self.W_ = U @ Vt
            else:
                if self.alpha == 0:
                    self.W_ = Y_centered.T @ np.linalg.pinv(X_centered.T)
                else:
                    XtX = X_centered.T @ X_centered
                    XtX_reg = XtX + self.alpha * np.eye(XtX.shape[0])
                    XtX_inv = np.linalg.inv(XtX_reg)
                    self.W_ = Y_centered.T @ X_centered @ XtX_inv

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned transformation to X.

        Parameters
        ----------
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.

        Returns
        -------
            X_proj: array of shape (n_samples, n_components)
                The transformed data in the low-dimensional space.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if hasattr(self, "_X_mean") and self._X_mean is not None:
            X_centered = X - self._X_mean
        else:
            X_centered = X
        X_proj: np.ndarray = (self.W_ @ X_centered.T).T
        return X_proj

    def _truncated_pinv(self, W: np.ndarray, tol: float = 1e-5) -> np.ndarray:
        U, S, VT = np.linalg.svd(W, full_matrices=False)
        S_inv = np.array([1 / s if s > tol else 0 for s in S])
        result: np.ndarray = VT.T @ np.diag(S_inv) @ U.T
        return result

    def _regularized_pinv(self, W: np.ndarray, lambda_: float = 1e-5) -> np.ndarray:
        result: np.ndarray = np.linalg.inv(W.T @ W + lambda_ * np.eye(W.shape[1])) @ W.T
        return result

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original input X from its low-dimensional projection.

        Parameters
        ----------
            X_proj: array-like of shape (n_samples, n_components)
                The low-dimensional representation of the input data.

        Returns
        -------
            X_reconstructed: array of shape (n_samples, original_n_features)
                The reconstructed data in the original space.
        """
        check_is_fitted(self)
        X_proj = check_array(X_proj, ensure_2d=True)

        # Use pseudo-inverse in case W_ is not square or full-rank
        # W_pinv = np.linalg.pinv(self.W_)
        # Use regularized pseudo-inverse to avoid numerical issues
        # W_pinv = self._regularized_pinv(self.W_)
        W_pinv = self._truncated_pinv(self.W_)

        X_centered: np.ndarray = (W_pinv @ X_proj.T).T

        if hasattr(self, "_X_mean") and self._X_mean is not None:
            result: np.ndarray = X_centered + self._X_mean
            return result
        else:
            return X_centered

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform X in one step.

        Parameters
        ----------
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2)

        Returns
        -------
            X_proj: array of shape (n_samples, n_components)
                The transformed data in the low-dimensional space.
        """
        result: np.ndarray = self.fit(X, y).transform(X)
        return result

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str | StressMetrics = StressMetrics.SCALE_NORMALIZED_STRESS,
    ) -> float:
        """Evaluate embedding quality using SUPERVISED metric (uses y labels)."""
        check_is_fitted(self)
        metric = self._validate_and_convert_metric(metric)
        X, y = self._validate_data(X, y, reset=False)
        D_ideal = self.stage_1_fitted_.compute_ideal_distances(y)

        # Compute predicted pairwise distances
        X_proj = self.transform(X)
        n = X_proj.shape[0]
        D_pred = np.linalg.norm(X_proj[:, np.newaxis, :] - X_proj[np.newaxis, :, :], axis=-1)

        if metric == StressMetrics.NORMALIZED_KL_DIVERGENCE:
            score_value = kl_divergence_stress(D_ideal, D_pred)
            score_value = float(exp(-score_value))
            return score_value

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask = mask & (D_ideal >= 0)
        D_ideal_flat = D_ideal[mask]
        D_pred_flat = D_pred[mask]

        if metric == StressMetrics.SCALE_NORMALIZED_STRESS:
            score_value = float(1 - scale_normalized_stress(D_ideal_flat, D_pred_flat))
        elif metric == StressMetrics.NON_METRIC_STRESS:
            score_value = float(1 - non_metric_stress(D_ideal_flat, D_pred_flat))
        elif metric == StressMetrics.SHEPARD_GOODNESS_SCORE:
            score_value = float(shepard_goodness_stress(D_ideal_flat, D_pred_flat))
        elif metric == StressMetrics.NORMALIZED_STRESS:
            score_value = float(1 - normalized_stress(D_ideal_flat, D_pred_flat))

        return score_value

    def save(self, filepath: str) -> None:
        """
        Save the model to disk, including learned weights.
        """
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "SupervisedMDS":
        """
        Load a model from disk.

        Returns
        -------
            An instance of SupervisedMDS.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
