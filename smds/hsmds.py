from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]
from sklearn.utils.validation import check_array, check_is_fitted, validate_data  # type: ignore[import-untyped]

from smds.smds import (
    ComputedSMDSParametrization,
    SupervisedMDS,
    UserProvidedSMDSParametrization,
)


class HybridSMDS(SupervisedMDS):
    """
    HybridSMDS: Combines supervised distance-matching from SupervisedMDS
    with a user-specified dimensionality reduction model.

    Supports 'bypass_mds' mode where 'y' is treated directly as target coordinates Y.
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
        if reducer is None:
            raise ValueError("HybridSMDS requires a reducer object (e.g. PCA, PLSRegression, etc.)")
        if bypass_mds:
            stage_1 = UserProvidedSMDSParametrization(np.zeros((2, n_components), dtype=np.float64), n_components)
        else:
            stage_1 = ComputedSMDSParametrization(manifold=manifold, n_components=n_components)
        super().__init__(
            stage_1=stage_1,
            alpha=alpha,
            orthonormal=orthonormal,
            radius=radius,
        )
        self.manifold = manifold
        self.reducer = reducer
        self.bypass_mds = bypass_mds

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "HybridSMDS":
        """
        Fit HybridSMDS.
        If bypass_mds=True, y is used as target embedding directly.
        If bypass_mds=False, y are labels used to compute MDS embedding.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.bypass_mds:
            X = check_array(X)
            if y.ndim != 2 or y.shape[1] != self.stage_1.n_components:
                raise ValueError(
                    f"When bypass_mds=True, y must be target coordinates with shape "
                    f"(n_samples, {self.stage_1.n_components}). Got shape {y.shape}."
                )
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
                )
            Y = y
        else:
            X, y = self._validate_data(X, y)
            distances = self.stage_1.compute_ideal_distances(y)
            if isinstance(distances, np.ndarray) and np.any(distances < 0):
                raise ValueError("HybridSMDS does not support incomplete distance matrices.")
            Y = self.stage_1._classical_mds(distances)
            n_comp = self.stage_1.n_components
            if Y.shape[1] < n_comp:
                pad = np.zeros((Y.shape[0], n_comp - Y.shape[1]), dtype=np.float64)
                Y = np.hstack([Y, pad])

        self.Y_ = Y
        self.stage_1_fitted_ = UserProvidedSMDSParametrization(Y, self.stage_1.n_components)
        self.stage_1_fitted_.fit()
        self.reducer_ = clone(self.reducer)
        self.reducer_.fit(X, self.Y_)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Projects X using the fitted reducer."""
        check_is_fitted(self, ["reducer_"])
        X = validate_data(self, X, reset=False)

        if not hasattr(self.reducer_, "transform"):
            raise RuntimeError("This reducer is not fitted or does not support transform.")
        X_proj: NDArray[np.float64] = self.reducer_.transform(X)
        return X_proj

    def inverse_transform(self, X_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        """Optional inverse projection, if the reducer supports it."""
        check_is_fitted(self, ["reducer_"])
        if not hasattr(self.reducer_, "inverse_transform"):
            raise NotImplementedError("This reducer does not support inverse_transform.")
        X_reconstructed: NDArray[np.float64] = self.reducer_.inverse_transform(X_proj)
        return X_reconstructed
