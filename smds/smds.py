import os
import pickle
from typing import Callable, Union

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin


class SupervisedMDS(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = 2,
        manifold: Union[str, Callable] = "trivial",
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
    ):
        """
        Parameters:
            n_components:
                Dimensionality of the target subspace.
            manifold:
                If 'trivial', 'cluster', 'discrete_circular', 'chain', y contains discrete values.
                If 'euclidean', 'linear', 'log_linear', 'circular', 'helix', 'semicircular',
                'log_semicircular', y contains continuous values.
                If 'sphere_chord', 'geodesic', 'cylinder_chord',
                y contains 2D coordinates (latitude, longitude).
                If callable, should return a (n x n) ideal distance matrix given y.
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
        n = len(y)
        D = np.zeros((n, n))

        if self.manifold in ["trivial", "cluster"]:  # Retrocompatibility
            D = (y[:, None] != y[None, :]).astype(float)
        elif self.manifold in ["euclidean", "linear"]:
            diff = y[None, :, None] - y[None, None, :]
            D = np.linalg.norm(diff, axis=0)
        elif self.manifold == "log_linear":
            log_y = np.log(y + 1)
            D = np.abs(log_y[:, None] - log_y[None, :])
        elif self.manifold == "helix":
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))  # shape: (n,)

            # Map to 3D spiral
            theta = 2 * np.pi * y_norm  # angle around the circle
            x = np.cos(theta)
            y_circle = np.sin(theta)
            z = y_norm  # vertical component

            spiral_coords = np.stack([x, y_circle, z], axis=1)  # shape: (n, 3)

            # Compute pairwise Euclidean distances in spiral space
            diffs = spiral_coords[:, None, :] - spiral_coords[None, :, :]
            D = np.linalg.norm(diffs, axis=2)  # shape: (n, n)

        elif self.manifold == "discrete_circular":
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    D[i, j] = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
        elif self.manifold == "chain":
            max_y = np.max(y)
            for i in range(n):
                for j in range(n):
                    dist = min(np.abs(y[i] - y[j]), max_y + 1 - np.abs(y[i] - y[j]))
                    D[i, j] = dist if dist < threshold else -1
        elif self.manifold == "semicircular":
            max_y = np.max(y)
            min_y = np.min(y)

            # Normalize y to [0, 1]
            y_norm = (y - min_y) / (max_y - min_y)

            # Pairwise absolute differences
            delta = np.abs(y_norm[:, None] - y_norm[None, :])
            D = 2 * np.sin((np.pi / 2) * delta)
        elif self.manifold == "log_semicircular":
            max_y = np.max(y)
            min_y = np.min(y)

            # Normalize y to [0, 1]
            y_norm = (y - min_y) / (max_y - min_y)

            # Log transform (add 1 to avoid log(0))
            y_log = np.log(y_norm + 1)

            # Pairwise absolute differences
            delta = np.abs(y_log[:, None] - y_log[None, :])
            D = 2 * np.sin((np.pi / 2) * delta)
        elif self.manifold == "sphere_chord":
            if len(y.shape) != 2 or y.shape[1] != 2:
                raise ValueError("For 'sphere_chord', labels must be a 2D array with shape (n_samples, 2).")
            lat_rad = np.radians(y[:, 0])
            lon_rad = np.radians(y[:, 1])
            radius = self.radius

            x = radius * np.cos(lat_rad) * np.cos(lon_rad)
            y_ = radius * np.cos(lat_rad) * np.sin(lon_rad)
            z = radius * np.sin(lat_rad)

            coords = np.stack([x, y_, z], axis=1)
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            D = np.linalg.norm(diffs, axis=2)

        elif self.manifold == "geodesic":
            if len(y.shape) != 2 or y.shape[1] != 2:
                raise ValueError("For 'geodesic', labels must be a 2D array with shape (n_samples, 2).")
            radius = self.radius
            lat = np.radians(y[:, 0])[:, np.newaxis]
            lon = np.radians(y[:, 1])[:, np.newaxis]

            dlat = lat - lat.T
            dlon = lon - lon.T

            lat1 = lat
            lat2 = lat.T

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            D = radius * c

        elif self.manifold == "cylinder_chord":
            if len(y.shape) != 2 or y.shape[1] != 2:
                raise ValueError("For 'cylinder_chord', labels must be a 2D array with shape (n_samples, 2).")

            lat_rad = np.radians(y[:, 0])  # latitude as height
            lon_rad = np.radians(y[:, 1])  # longitude as angle

            radius = self.radius  # cylinder radius

            x = radius * np.cos(lon_rad)
            y_ = radius * np.sin(lon_rad)
            z = lat_rad  # treat lat as height

            coords = np.stack([x, y_, z], axis=1)
            diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            D = np.linalg.norm(diffs, axis=2)

        elif callable(self.manifold):
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear transformation W to match distances induced by labels y.
        Uses classical MDS + closed-form when all distances are defined,
        and switches to optimization if some distances are undefined (negative).

        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2)
                The labels or coordinates defining the ideal distances.
        Returns:
            self: returns an instance of self.
        """
        X = np.asarray(X)
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
        else:
            # Use classical MDS + closed-form least squares
            Y = self._classical_mds(D)
            self.Y_ = Y

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

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform X in one step.
        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2)
                The labels or coordinates defining the ideal distances.
        Returns:
            X_proj: array of shape (n_samples, n_components)
                The transformed data in the low-dimensional space.
        """
        return self.fit(X, y).transform(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute how well the transformed distances match ideal distances.

        Parameters:
            X: array-like of shape (n_samples, n_features)
                The input data to be transformed.
            y: array-like of shape (n_samples,) or (n_samples, 2)
                The labels or coordinates defining the ideal distances.
        Returns:
            score: A score between -∞ and 1. Higher is better.
        """
        if self.W_ is None:
            raise RuntimeError("Model must be fit before scoring.")

        D_true = self._compute_ideal_distances(y)
        X_proj = self.transform(X)

        # Compute predicted pairwise distances
        n = X_proj.shape[0]
        D_pred = np.linalg.norm(X_proj[:, np.newaxis, :] - X_proj[np.newaxis, :, :], axis=-1)

        # Compute stress and normalize
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        stress = np.sum((D_pred[mask] - D_true[mask]) ** 2)
        denom = np.sum(D_true[mask] ** 2)

        score = 1 - stress / denom if denom > 0 else -np.inf

        return score

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
