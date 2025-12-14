import csv
import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.model_selection import KFold  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape
from smds.shapes.continuous_shapes.circular import CircularShape
from smds.shapes.continuous_shapes.euclidean import EuclideanShape
from smds.shapes.continuous_shapes.log_linear import LogLinearShape
from smds.shapes.continuous_shapes.semicircular import SemicircularShape
from smds.shapes.discrete_shapes.chain import ChainShape
from smds.shapes.discrete_shapes.cluster import ClusterShape
from smds.shapes.discrete_shapes.discrete_circular import DiscreteCircularShape
from smds.shapes.spatial_shapes.cylindrical import CylindricalShape
from smds.shapes.spatial_shapes.geodesic import GeodesicShape
from smds.shapes.spatial_shapes.spherical import SphericalShape
from smds.shapes.spiral_shape import SpiralShape
from smds.smds import SupervisedMDS


class ManifoldDiscovery(BaseEstimator):  # type: ignore[misc]
    """
    Manifold Discovery tool to evaluate different manifolds using SMDS and k-fold cross-validation.
    """

    def __init__(
        self,
        manifolds: Optional[List[BaseShape]] = None,
        k_folds: int = 10,
        random_state: int = 42,
        save_path: str = "manifold_discovery_results",
        n_components: int = 2,
        cleanup_cache: bool = True,
        n_jobs: int = -1,
    ):
        """
        Initialize the ManifoldDiscovery.

        Parameters:
            manifolds: List of shape instances to evaluate. If None, uses a default set.
            k_folds: Number of folds for cross-validation.
            random_state: Seed for reproducibility.
            save_path: Directory to save intermediate and final results.
            n_components: Number of components for SMDS.
            cleanup_cache: If True, deletes cache files after successful completion.
            n_jobs: Number of parallel jobs. -1 means use all available CPUs.
        """
        if manifolds is None:
            manifolds = [
                ChainShape(),
                ClusterShape(),
                DiscreteCircularShape(),
                CircularShape(),
                CylindricalShape(),
                GeodesicShape(),
                SphericalShape(),
                SpiralShape(),
                LogLinearShape(),
                EuclideanShape(),
                SemicircularShape(),
            ]
        self.manifolds = manifolds
        self.k_folds = k_folds
        self.random_state = random_state
        self.save_path = save_path
        self.n_components = n_components
        self.cleanup_cache = cleanup_cache
        self.n_jobs = n_jobs
        self.results_: List[Dict[str, Any]] = []
        self.best_estimator_: Optional[SupervisedMDS] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -np.inf
        self.best_manifold_name_: Optional[str] = None

    def _hash_data(self, X: NDArray[np.float64]) -> str:
        """
        Computes a hash of the input data X.
        This is done once at the beginning to avoid repeated hashing.
        """
        if not X.flags["C_CONTIGUOUS"]:
            X = np.ascontiguousarray(X)
        hasher = hashlib.md5(X.view(np.uint8))
        return hasher.hexdigest()

    def _compute_hash(
        self,
        manifold_name: str,
        params: Dict[str, Any],
        fold_idx: int,
        X_hash: str,
    ) -> str:
        """
        Computes a unique hash for a run configuration.
        Uses pre-computed X_hash to avoid repeated hashing of large arrays.
        """
        config_str: str = f"{manifold_name}_{json.dumps(params, sort_keys=True)}_{fold_idx}_{X_hash}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_cached_result(self, result_file: str) -> Optional[float]:
        """Load a cached result from disk. Returns score if found, None otherwise."""
        if not os.path.exists(result_file):
            return None
        try:
            with open(result_file, "rb") as f:
                result_data: Dict[str, Any] = pickle.load(f)
            score = result_data.get("score")
            return float(score) if score is not None else None
        except (pickle.UnpicklingError, KeyError, OSError, ValueError, TypeError):
            return None

    def _save_result(
        self,
        result_file: str,
        manifold_name: str,
        params: Dict[str, Any],
        fold_idx: int,
        score: float,
        run_hash: str,
    ) -> None:
        """Save a result to disk."""
        result_data = {
            "manifold": manifold_name,
            "params": params,
            "fold": fold_idx,
            "score": score,
            "hash": run_hash,
        }
        with open(result_file, "wb") as f:
            pickle.dump(result_data, f)

    def _process_fold(
        self,
        manifold: BaseShape,
        manifold_name: str,
        params: Dict[str, Any],
        fold_idx: int,
        train_index: np.ndarray,
        test_index: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        X_hash: str,
    ) -> Tuple[float, str]:
        """
        Process a single fold. Returns (score, cache_file_path).
        """
        run_hash = self._compute_hash(manifold_name, params, fold_idx, X_hash)
        result_file = os.path.join(self.save_path, f"{run_hash}.pkl")

        cached_score = self._load_cached_result(result_file)
        if cached_score is not None:
            return cached_score, result_file

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        smds = SupervisedMDS(manifold=manifold, n_components=self.n_components)

        try:
            smds.fit(X_train, y_train)
            score = smds.score(X_test, y_test)
            self._save_result(result_file, manifold_name, params, fold_idx, score, run_hash)
            return score, result_file
        except Exception:
            return np.nan, result_file

    def _process_manifold_results(
        self,
        manifold_name: str,
        params: Dict[str, Any],
        manifold: BaseShape,
        results: List[Tuple[float, str]],
        cache_files: List[str],
    ) -> None:
        """Process and store results for a single manifold."""
        fold_scores = []
        for fold_idx, (score, result_file) in enumerate(results):
            fold_scores.append(score)
            if result_file not in cache_files:
                cache_files.append(result_file)
            if not np.isnan(score):
                print(f"  Fold {fold_idx + 1}/{self.k_folds}: Score: {score:.4f}")
            else:
                print(f"  Fold {fold_idx + 1}/{self.k_folds}: Failed")

        avg_score = np.nanmean(fold_scores)
        self.results_.append(
            {
                "manifold": manifold_name,
                "params": params,
                "fold_scores": fold_scores,
                "average_score": avg_score,
                "estimator": manifold,
            }
        )

    def _select_best_estimator(self, X: np.ndarray, y: np.ndarray) -> None:
        """Select and refit the best estimator based on results."""
        valid_results = [r for r in self.results_ if not np.isnan(r["average_score"])]

        if not valid_results:
            print("No valid results found (all scores NaN or failures).")
            return

        valid_results.sort(key=lambda x: x["average_score"], reverse=True)
        best_result = valid_results[0]

        self.best_score_ = best_result["average_score"]
        self.best_params_ = best_result["params"]
        self.best_manifold_name_ = best_result["manifold"]

        print(f"\nRefitting best model: {self.best_manifold_name_} with score {self.best_score_:.4f}")

        best_manifold = best_result["estimator"]
        self.best_estimator_ = SupervisedMDS(manifold=best_manifold, n_components=self.n_components)
        self.best_estimator_.fit(X, y)

    def _cleanup_cache_files(self, cache_files: List[str]) -> None:
        """Remove cache files if cleanup is enabled."""
        if not self.cleanup_cache:
            return

        for cache_file in cache_files:
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            except OSError:
                pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManifoldDiscovery":
        """
        Perform manifold discovery.

        Iterates over all manifolds and k-folds. Checks for existing results on disk
        to avoid re-computation (resume capability). Uses parallel processing for folds.

        Parameters:
            X: Input features.
            y: Input labels/targets.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        X_hash = self._hash_data(X)
        cache_files: List[str] = []

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        fold_splits = list(kf.split(X))

        for manifold in self.manifolds:
            manifold_name = manifold.__class__.__name__
            params = manifold.get_params()

            print(f"Processing manifold: {manifold_name} with params: {params}")

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._process_fold)(
                    manifold,
                    manifold_name,
                    params,
                    fold_idx,
                    train_index,
                    test_index,
                    X,
                    y,
                    X_hash,
                )
                for fold_idx, (train_index, test_index) in enumerate(fold_splits)
            )

            self._process_manifold_results(manifold_name, params, manifold, results, cache_files)

        self._save_summary()
        self._select_best_estimator(X, y)
        self._cleanup_cache_files(cache_files)

        return self

    def _save_summary(self) -> None:
        """Saves a summary of all results to a CSV file."""

        summary_file = os.path.join(self.save_path, "summary.csv")

        csv_rows = []
        for res in self.results_:
            row = {
                "manifold": res["manifold"],
                "average_score": res["average_score"],
                "params": json.dumps(res["params"]),
            }
            for i, score in enumerate(res["fold_scores"]):
                row[f"fold_{i}_score"] = score
            csv_rows.append(row)

        if not csv_rows:
            return

        fieldnames = ["manifold", "average_score", "params"] + [f"fold_{i}_score" for i in range(self.k_folds)]

        with open(summary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Summary saved to {summary_file}")

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results_

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Call transform on the best found estimator.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model is not fitted or no valid manifold found.")
        return self.best_estimator_.transform(X)

    def fit_transform(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Fit to data, then transform it.
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_proj: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Call inverse_transform on the best found estimator.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model is not fitted or no valid manifold found.")
        return self.best_estimator_.inverse_transform(X_proj)

    def score(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """
        Call score on the best found estimator.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("Model is not fitted or no valid manifold found.")
        return self.best_estimator_.score(X, y)
