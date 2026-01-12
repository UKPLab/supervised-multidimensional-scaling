import os
import pickle
from abc import ABC, abstractmethod
from math import exp
from typing import Callable

import numpy as np
from scipy.linalg import eigh  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin, clone  # type: ignore[import-untyped]
from sklearn.utils.multiclass import type_of_target  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

from smds.stress import (
    StressMetrics,
    kl_divergence_stress,
    non_metric_stress,
    normalized_stress,
    scale_normalized_stress,
    shepard_goodness_stress,
)

# TODO: stage 2 - for mapping to the lower space
#   result: X_proj


# smds stage 1 - for the manifold Y_
class Stage1SMDSTransformer(TransformerMixin, BaseEstimator, ABC):
    @property
    @abstractmethod
    def n_components(self) -> int:
        """
        Number of components of the projected manifold.
        """
        pass

    @abstractmethod
    def fit(self, X, y=None):
        """
        Subclasses must implement this.
        It is required for TransformerMixin.fit_transform to work.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Subclasses must implement this.
        It is required for TransformerMixin.fit_transform to work.
        """
        pass

    @abstractmethod
    def compute_ideal_distances(self, y: np.typing.NDArray):
        pass


class ComputedStage1(Stage1SMDSTransformer):
    def __init__(self, manifold: Callable[[np.ndarray], np.ndarray], n_components: int):
        # fixme: set manifold to be BaseShape
        self.manifold = manifold
        self._n_components = n_components

    @property
    def n_components(self) -> int:
        return self._n_components

    def compute_ideal_distances(self, y: np.typing.NDArray, threshold: int = 2) -> np.ndarray:
        """
        Compute ideal pairwise distance matrix D based on labels y and specified self.manifold.
        """
        if callable(self.manifold):
            D: np.ndarray = self.manifold(y)
        else:
            raise ValueError("Invalid manifold specification.")

        return D

    def _classical_mds(self, D: np.ndarray) -> np.ndarray:
        """
        Perform Classical MDS on the distance matrix D to obtain a low-dimensional embedding.
        This is the template manifold for the supervised MDS.
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

    def fit(self, X, y=None) -> "ComputedStage1":
        self.D_ = self.compute_ideal_distances(X)
        self.Y_ = self._classical_mds(self.D_)
        return self

    def transform(self, X=None) -> np.typing.NDArray:
        return self.Y_


class UserProvidedStage1(Stage1SMDSTransformer):
    def __init__(self, y: np.typing.NDArray, n_components: int):
        self._n_components = n_components
        if y.shape[-1] != n_components:
            raise ValueError(f"y must have last shape == n_components ({n_components}), got {y.shape[-1]}")
        # todo: maybe no need to pass it here if we pass it in fit()?
        self.y = y

    @property
    def n_components(self) -> int:
        return self._n_components

    def compute_ideal_distances(self, y: np.typing.NDArray):
        return np.linalg.norm(self.Y_[:, np.newaxis, :] - self.Y_[np.newaxis, :, :], axis=-1)

    def fit(self, X=None, y=None) -> "UserProvidedStage1":
        self.Y_ = self.y
        self.D_ = self.compute_ideal_distances(None)
        return self

    def transform(self, X):
        return self.Y_


class SupervisedMDS(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    def __init__(
        self,
        stage_1: Stage1SMDSTransformer,
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
    ):
        """
        Parameters
        ----------
            stage_1:
                Stage 1 of SMDS.
                Defines the method of obtaining Y_.
            metric:
                The metric to use for scoring the embedding.
        """
        self.stage_1 = stage_1
        self.alpha = alpha
        self.orthonormal = orthonormal
        self.radius = radius

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
        W = W_flat.reshape((self.stage_1.n_components, X.shape[1]))
        X_proj = (W @ X.T).T
        D_pred = np.linalg.norm(X_proj[:, None, :] - X_proj[None, :, :], axis=-1)
        loss = (D_pred - D)[mask]
        result: float = float(np.sum(loss**2))
        return result

    def _validate_data(self, X: np.ndarray, y: np.ndarray, reset: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and process X and y based on the manifold's expected y dimensionality.
        """
        if isinstance(self.stage_1, UserProvidedStage1):
            # todo: probably not the right way
            expected_y_ndim = y.ndim
        else:
            expected_y_ndim = getattr(self.stage_1.manifold, "y_ndim", 1)

        if expected_y_ndim == 1:
            X, y = validate_data(self, X, y, reset=reset)
            type_of_target(y, raise_unknown=True)
            y = np.asarray(y).squeeze()
            if y.ndim == 0:
                y = y.reshape(1)
        else:
            X = check_array(X)
            y = np.asarray(y)
            if y.ndim != expected_y_ndim:
                raise ValueError(
                    f"Input 'y' must be {expected_y_ndim}-dimensional, "
                    f"but got shape {y.shape} with {y.ndim} dimensions."
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
                )

        return X, y

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
        X, y = self._validate_data(X, y)

        if X.shape[0] == 1:
            raise ValueError("Found array with n_samples=1. SupervisedMDS requires at least 2 samples.")
        if self.orthonormal and self.alpha != 0:
            print("Warning: orthonormal=True and alpha!=0. alpha will be ignored.")

        X = np.asarray(X)
        y = np.asarray(y).squeeze()  # Ensure y is 1D

        self.stage_1_fitted_: Stage1SMDSTransformer = clone(self.stage_1)
        self.stage_1_fitted_.fit(y)

        self.Y_ = self.stage_1_fitted_.Y_

        D = self.stage_1_fitted_.D_

        if np.any(D < 0):
            # Raise warning if any distances are negative
            print("Warning: Distance matrix is incomplete. Using optimization to fit W.")
            mask = D >= 0
            rng = np.random.default_rng(42)
            W0 = rng.normal(scale=0.01, size=(self.stage_1_fitted_.n_components, X.shape[1]))

            result = minimize(self._masked_loss, W0.ravel(), args=(X, D, mask), method="L-BFGS-B")
            self.W_ = result.x.reshape((self.stage_1_fitted_.n_components, X.shape[1]))
        else:
            # Use classical MDS + closed-form least squares
            Y = self.Y_

            self._X_mean = X.mean(axis=0)  # Centering
            self._Y_mean = Y.mean(axis=0)  # Centering Y
            X_centered = X - X.mean(axis=0)
            Y_centered = Y - Y.mean(axis=0)
            if self.orthonormal:
                # Orthogonal Procrustes
                M = Y_centered.T @ X_centered
                U, _, Vt = np.linalg.svd(M)
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
