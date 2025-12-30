from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from smds import SupervisedMDS


class HybridSMDS(SupervisedMDS):
    """
    HybridSMDS: Allows explicit separation between target generation and mapping learning.

    Supports Issue #53/#65:
    If 'y' passed to fit() has shape (n_samples, n_components), it is treated directly
    as the target embedding Y, bypassing the internal MDS step.
    """

    def __init__(
        self,
        manifold: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_components: int = 2,
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
        reducer: Optional[BaseEstimator] = None,
        bypass_mds: bool = False,
    ):
        super().__init__(
            manifold=manifold, n_components=n_components, alpha=alpha, orthonormal=orthonormal, radius=radius
        )

        if reducer is None:
            raise ValueError("HybridSMDS requires a reducer object (e.g. PLSRegression).")

        self.reducer = reducer
        self.bypass_mds = bypass_mds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "HybridSMDS":
        """
        Fit HybridSMDS by computing ideal distances (via SupervisedMDS) and fitting
        the reducer so its output approximates classical MDS embeddings.

        Parameters:
            X: Input data (n_samples, n_features).
            y: Target information.
               - If shape is (n_samples,): Treated as labels. Ideal distances are computed,
                 and MDS generates Y.
               - If shape is (n_samples, n_components): Treated directly as target coordinates Y.
                 MDS step is skipped (Direct Input Mode).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.bypass_mds:
            if y.ndim != 2 or y.shape[1] != self.n_components:
                raise ValueError(
                    f"When bypass_mds=True, y must be target coordinates with shape "
                    f"(n_samples, {self.n_components}). Got shape {y.shape}."
                )
            Y = y
            self.Y_ = Y

        else:
            if hasattr(self.manifold, "y_ndim") and self.manifold.y_ndim == 1 and y.ndim > 1:
                y = y.squeeze()

            distances = self._compute_ideal_distances(y)

            if isinstance(distances, np.ndarray) and np.any(distances < 0):
                raise ValueError("HybridSMDS: does not support incomplete distance matrices.")

            Y = self._classical_mds(distances)
            self.Y_ = Y

        self.reducer.fit(X, Y)

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not hasattr(self.reducer, "transform"):
            raise RuntimeError("This reducer is not fitted or does not support transform.")
        X_proj: NDArray[np.float64] = self.reducer.transform(X)
        return X_proj

    def inverse_transform(self, X_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        if not hasattr(self.reducer, "inverse_transform"):
            raise NotImplementedError("This reducer does not support inverse_transform.")
        X_reconstructed: NDArray[np.float64] = self.reducer.inverse_transform(X_proj)
        return X_reconstructed
