from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore[import-untyped]

from smds.smds import SMDSParametrization, SupervisedMDS, UserProvidedSMDSParametrization


class HybridSMDS(SupervisedMDS):
    """
    Combines MDS manifold construction with a custom dimensionality reduction model.

    Uses a user-provided reducer (PLSRegression, PCA, etc.) instead of linear projection
    to map high-dimensional data onto the MDS embedding.

    Parameters
    ----------
    manifold : str or Callable, default="circular"
        Manifold type ("circular", "spherical", etc.) or custom distance function.
    n_components : int, default=2
        Target embedding dimensions.
    reducer : BaseEstimator, required
        sklearn-compatible reducer with fit(X, Y) and transform(X).
    bypass_mds : bool, default=False
        If True, treat y as target coordinates directly.
    alpha, orthonormal, radius : inherited from SupervisedMDS
    """

    def __init__(
        self,
        manifold: Union[str, Callable[[NDArray[Any]], NDArray[np.float64]]] = "circular",
        n_components: int = 2,
        alpha: float = 0.1,
        orthonormal: bool = False,
        radius: float = 6371,
        reducer: BaseEstimator | None = None,
        bypass_mds: bool = False,
    ):
        arr_bypass = np.asarray(bypass_mds)
        if arr_bypass.size == 0:
            bypass_mds_bool = False
        elif arr_bypass.size == 1:
            bypass_mds_bool = bool(arr_bypass.item())
        else:
            bypass_mds_bool = bool(arr_bypass.flat[0])

        if bypass_mds_bool:
            stage_1: SMDSParametrization | str = "user_provided"
            manifold_for_super = "circular"
        elif isinstance(manifold, str):
            stage_1 = "computed"
            manifold_for_super = manifold
        else:
            from smds.smds import ComputedSMDSParametrization

            stage_1 = ComputedSMDSParametrization(manifold=manifold, n_components=n_components)
            manifold_for_super = "circular"

        super().__init__(
            stage_1=stage_1,
            manifold=manifold_for_super,
            alpha=alpha,
            orthonormal=orthonormal,
            radius=radius,
        )
        self.n_components = n_components
        self.manifold = manifold
        self.reducer = reducer
        self.bypass_mds = bypass_mds

    def _validate_data(
        self, X: np.ndarray, y: np.ndarray, reset: bool = True, stage_1_model: SMDSParametrization | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.utils.validation import check_array

        arr_bypass = np.asarray(self.bypass_mds)
        if arr_bypass.size == 0:
            bypass_bool = False
        elif arr_bypass.size == 1:
            bypass_bool = bool(arr_bypass.item())
        else:
            bypass_bool = bool(arr_bypass.flat[0])

        if bypass_bool:
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
            return super()._validate_data(X, y, reset=reset, stage_1_model=stage_1_model)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "HybridSMDS":
        if self.reducer is None:
            raise ValueError("HybridSMDS requires a reducer object (e.g. PCA, PLSRegression, etc.)")

        X, y = self._validate_data(X, y)

        if self.bypass_mds:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            Y = y_arr
        else:
            stage_1_model = self._resolve_stage_1()
            stage_1_fitted = clone(stage_1_model)
            stage_1_fitted.fit(y)

            try:
                distances = stage_1_fitted.D_
            except AttributeError:
                try:
                    distances = stage_1_fitted.compute_ideal_distances(y)
                except (ValueError, TypeError) as e:
                    if "dtype" in str(e).lower() or "type" in str(e).lower():
                        from sklearn.utils.multiclass import type_of_target

                        try:
                            type_of_target(y, raise_unknown=True)
                        except ValueError:
                            raise ValueError("Unknown label type: object") from e
                    raise

            if isinstance(distances, np.ndarray) and np.any(distances < 0):
                raise ValueError("HybridSMDS does not support incomplete distance matrices.")

            Y = stage_1_fitted.Y_

            if Y.shape[1] < self.n_components:
                pad = np.zeros((Y.shape[0], self.n_components - Y.shape[1]), dtype=np.float64)
                Y = np.hstack([Y, pad])

            self.stage_1_fitted_ = stage_1_fitted

        self.Y_ = Y

        if self.bypass_mds:
            self.stage_1_fitted_ = UserProvidedSMDSParametrization(y=Y, n_components=self.n_components)
            self.stage_1_fitted_.fit()

        self.reducer_ = clone(self.reducer)
        self.reducer_.fit(X, self.Y_)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, ["reducer_", "Y_"])
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
