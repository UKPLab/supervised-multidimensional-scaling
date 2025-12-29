import os
import pickle
from typing import Callable, Optional, Union

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin

from smds.stress.non_metric_stress import NonMetricStress
from smds.stress.scale_normalized_stress import ScaleNormalizedStress
from smds.stress.stress_metrics import StressMetrics


class SupervisedMDS(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        manifold: Optional[Callable] = None,
        n_components: int = 2,
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
    ):
        """
        Parameters:
            n_components:
                Dimensionality of the target subspace.
            manifold:
                If callable, should return a (n x n) ideal distance matrix given y.
                Optional when using direct embedding (Y parameter in fit).
        """
        self.n_components = n_components
        self.manifold = manifold
        self.W_ = None
        self.alpha = alpha
        self.orthonormal = orthonormal
        self.radius = radius  # Only used for spherical manifolds
        self._X_mean = None
        self._Y_mean = None
        if orthonormal and alpha != 0:
            print("Warning: orthonormal=True and alpha!=0. alpha will be ignored.")

    def _compute_ideal_distances(self, y: np.ndarray, threshold: int = 2) -> np.ndarray:
        """
        Compute ideal pairwise distance matrix D based on labels y and specified self.manifold.
        """
        if self.manifold is None:
            raise ValueError(
                "Manifold is not set. Either provide a manifold in __init__ or use "
                "direct embedding (Y parameter) in fit()."
            )
        if callable(self.manifold):
            D = self.manifold(y)
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
        Y = eigvecs * np.sqrt(np.maximum(eigvals, 0))
        return Y

    def _masked_loss(self, W_flat: np.ndarray, X: np.ndarray, D: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute the loss only on the defined distances (where mask is True).
        """
        W = W_flat.reshape((self.n_components, X.shape[1]))
        X_proj = (W @ X.T).T
        D_pred = np.linalg.norm(X_proj[:, None, :] - X_proj[None, :, :], axis=-1)
        loss = (D_pred - D)[mask]
        return np.sum(loss**2)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ):
        """
        Fit the linear transformation W to match distances induced by labels y.
        If Y is provided, it will be used to directly match a provided target embedding Y.
        If Y is not provided, it will be computed from the labels y using the manifold.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2), optional
                The labels or coordinates defining the ideal distances.
                Required if Y is not provided.
            Y: array-like of shape (n_samples, n_components), optional
                Direct target embedding to match. If provided, bypasses the
                manifold distance computation and MDS step entirely.
                This allows using pre-computed embeddings (e.g., from EmbeddingBuilder).
        Returns:
            self: returns an instance of self.
        """
        X = np.asarray(X)

        if Y is not None and y is not None:
            raise ValueError("Either Y (target embedding) or y (labels) must be provided, not both.")
        if Y is None and y is None:
            raise ValueError("Either Y (target embedding) or y (labels) must be provided.")

        # Determine target embedding Y
        if Y is not None:
            # Direct embedding mode: use provided Y, skip manifold and MDS
            Y = np.asarray(Y)
            if Y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Y must have same number of samples as X. "
                    f"Got Y.shape[0]={Y.shape[0]}, X.shape[0]={X.shape[0]}"
                )
            if Y.shape[1] != self.n_components:
                raise ValueError(
                    f"Y must have same number of components as n_components. "
                    f"Got Y.shape[1]={Y.shape[1]}, n_components={self.n_components}"
                )
            self.Y_ = Y
        elif y is not None:
            # Standard mode: compute Y from labels via manifold + MDS
            y = np.asarray(y).squeeze()  # Ensure y is 1D
            D = self._compute_ideal_distances(y)

            if np.any(D < 0):
                # Raise warning if any distances are negative
                print("Warning: Distance matrix is incomplete. Using optimization to fit W.")
                mask = D >= 0
                rng = np.random.default_rng(42)
                W0 = rng.normal(scale=0.01, size=(self.n_components, X.shape[1]))

                result = minimize(self._masked_loss, W0.ravel(), args=(X, D, mask), method="L-BFGS-B")
                self.W_ = result.x.reshape((self.n_components, X.shape[1]))
                return self

            Y = self._classical_mds(D)
            self.Y_ = Y

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

        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
        Returns:
            X_proj: array of shape (n_samples, n_components)
                The transformed data in the low-dimensional space.
        """
        if self.W_ is None:
            raise RuntimeError("You must fit the model before calling transform.")
        X = np.asarray(X)
        if self._X_mean is not None:
            # Center X using the same logic as during fit
            X_centered = X - self._X_mean
        else:
            X_centered = X
        X_proj = (self.W_ @ X_centered.T).T
        return X_proj

    def _truncated_pinv(self, W: np.ndarray, tol: float = 1e-5) -> np.ndarray:
        U, S, VT = np.linalg.svd(W, full_matrices=False)
        S_inv = np.array([1 / s if s > tol else 0 for s in S])
        return VT.T @ np.diag(S_inv) @ U.T

    def _regularized_pinv(self, W: np.ndarray, lambda_: float = 1e-5) -> np.ndarray:
        return np.linalg.inv(W.T @ W + lambda_ * np.eye(W.shape[1])) @ W.T

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original input X from its low-dimensional projection.

        Parameters:
            X_proj: array-like of shape (n_samples, n_components)
                The low-dimensional representation of the input data.

        Returns:
            X_reconstructed: array of shape (n_samples, original_n_features)
                The reconstructed data in the original space.
        """
        if self.W_ is None:
            raise RuntimeError("You must fit the model before calling inverse_transform.")

        X_proj = np.asarray(X_proj)

        # Use pseudo-inverse in case W_ is not square or full-rank
        # W_pinv = np.linalg.pinv(self.W_)
        # Use regularized pseudo-inverse to avoid numerical issues
        # W_pinv = self._regularized_pinv(self.W_)
        W_pinv = self._truncated_pinv(self.W_)

        X_centered = (W_pinv @ X_proj.T).T

        if hasattr(self, "_X_mean") and self._X_mean is not None:
            return X_centered + self._X_mean
        else:
            return X_centered

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit the model and transform X in one step.
        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2), optional
                The labels defining ideal distances. Required if Y not provided.
            Y: array-like of shape (n_samples, n_components), optional
                Direct target embedding to match.
        Returns:
            X_proj: array of shape (n_samples, n_components)
                The transformed data in the low-dimensional space.
        """
        return self.fit(X, y=y, Y=Y).transform(X)

    def score(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        metric: StressMetrics = StressMetrics.SCALE_NORMALIZED_STRESS,
    ) -> float:
        """
        Evaluate embedding quality using SUPERVISED metric.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data.
            y: array-like, optional
                Labels to compute ideal distances via manifold.
            Y: array-like of shape (n_samples, n_components), optional
                Direct target embedding. If provided, ideal distances are
                computed as Euclidean distances between Y points.
            metric: StressMetrics
                The stress metric to use for evaluation.

        Returns:
            float: Score value (1 - stress), higher is better.
        """
        if self.W_ is None:
            raise RuntimeError("Model must be fit before scoring.")

        # Compute ideal distances
        if Y is not None:
            # Direct embedding mode: Euclidean distances in Y
            Y = np.asarray(Y)
            n = Y.shape[0]
            D_ideal = np.linalg.norm(Y[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=-1)
        elif y is not None:
            # Standard mode: distances from manifold
            D_ideal = self._compute_ideal_distances(y)
            n = len(y)
        else:
            raise ValueError("Either y (labels) or Y (target embedding) must be provided.")

        # Compute predicted pairwise distances
        X_proj = self.transform(X)
        D_pred = np.linalg.norm(X_proj[:, np.newaxis, :] - X_proj[np.newaxis, :, :], axis=-1)

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask = mask & (D_ideal >= 0)
        D_ideal_flat = D_ideal[mask]
        D_pred_flat = D_pred[mask]

        # Compute stress
        if metric == StressMetrics.SCALE_NORMALIZED_STRESS:
            score_value = 1 - ScaleNormalizedStress().compute(D_ideal_flat, D_pred_flat)
        elif metric == StressMetrics.NON_METRIC_STRESS:
            score_value = 1 - NonMetricStress().compute(D_ideal_flat, D_pred_flat)
        # TODO: Add other metrics from the paper here
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return score_value

    def save(self, filepath: str):
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
        Returns:
            An instance of SupervisedMDS.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
