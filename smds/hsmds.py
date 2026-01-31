from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore[import-untyped]

from smds.smds import (
    ComputedSMDSParametrization,
    SupervisedMDS,
    UserProvidedSMDSParametrization,
)


def _resolve_stage_1(
    manifold: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_components: object,
    bypass_mds: object,
) -> Tuple[object, int, bool]:
    n_comp = 2
    if np.isscalar(n_components) and isinstance(n_components, (int, float)):
        n = int(n_components)
        if n > 0:
            n_comp = n
    arr_b = np.asarray(bypass_mds)
    bypass = bool(bypass_mds) if np.isscalar(bypass_mds) else (bool(arr_b.flat[0]) if arr_b.size else False)
    if bypass:
        stage_1 = UserProvidedSMDSParametrization(
            np.zeros((2, n_comp), dtype=np.float64), n_comp
        )
    else:
        manifold_fn = (
            manifold
            if callable(manifold)
            else (lambda y: np.zeros((len(y), len(y)), dtype=np.float64))
        )
        stage_1 = ComputedSMDSParametrization(manifold=manifold_fn, n_components=n_comp)
    return stage_1, n_comp, bypass


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
        stage_1, n_comp, bypass = _resolve_stage_1(manifold, n_components, bypass_mds)
        super().__init__(
            stage_1=stage_1,
            alpha=alpha,
            orthonormal=orthonormal,
            radius=radius,
        )
        self.n_components = n_comp
        self.manifold = manifold
        self.reducer = reducer
        self.bypass_mds = bypass

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "HybridSMDS":
        """
        Fit HybridSMDS.
        If bypass_mds=True, y is used as target embedding directly.
        If bypass_mds=False, y are labels used to compute MDS embedding.
        """
        if self.reducer is None:
            raise ValueError(
                "HybridSMDS requires a reducer object (e.g. PCA, PLSRegression, etc.)"
            )
        X, y = self._validate_data(X, y)

        if self.bypass_mds:
            Y = y
        else:
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
