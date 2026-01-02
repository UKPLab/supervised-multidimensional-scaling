from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

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
        if self.reducer is None:
            raise ValueError("HybridSMDS requires a reducer object (e.g. PLSRegression).")

        self.reducer_ = clone(self.reducer)

        if self.bypass_mds:
            X = check_array(X)
            y = np.asarray(y)
            if y.ndim != 2 or y.shape[1] != self.n_components:
                raise ValueError(
                    f"When bypass_mds=True, y must be target coordinates with shape "
                    f"(n_samples, {self.n_components}). Got shape {y.shape}."
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
                )
            self.n_features_in_ = X.shape[1]
            Y = y
            self.Y_ = Y
        else:
            X, y = self._validate_data(X, y)
            distances = self._compute_ideal_distances(y)

            if isinstance(distances, np.ndarray) and np.any(distances < 0):
                raise ValueError("HybridSMDS: does not support incomplete distance matrices.")

            Y = self._classical_mds(distances)
            self.Y_ = Y

        self.reducer_.fit(X, Y)

        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, ["reducer_"])
        X = validate_data(self, X, reset=False)

        if not hasattr(self.reducer_, "transform"):
            raise RuntimeError("This reducer is not fitted or does not support transform.")
        X_proj: NDArray[np.float64] = self.reducer_.transform(X)
        return X_proj

    def inverse_transform(self, X_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, ["reducer_"])

        if not hasattr(self.reducer_, "inverse_transform"):
            raise NotImplementedError("This reducer does not support inverse_transform.")
        X_reconstructed: NDArray[np.float64] = self.reducer_.inverse_transform(X_proj)
        return X_reconstructed
