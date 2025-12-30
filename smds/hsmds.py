from typing import Union, Callable, Optional
import numpy as np
from sklearn.base import BaseEstimator
from smds import SupervisedMDS
from numpy.typing import NDArray

class HybridSMDS(SupervisedMDS):
    """
    HybridSMDS: Allows explicit separation between target generation and mapping learning.
    
    Supports Issue #53/#65: 
    If 'y' passed to fit() has shape (n_samples, n_components), it is treated directly 
    as the target embedding Y, bypassing the internal MDS step.
    """

    def __init__(self,
                 manifold: Callable[[NDArray[np.float64]], NDArray[np.float64]],
                 n_components: int = 2,
                 alpha: float = 0.1,
                 orthonormal: bool = False,
                 radius: float = 6371,
                 reducer: Optional[BaseEstimator] = None):
        
        super().__init__(manifold=manifold,
                         n_components=n_components,
                         alpha=alpha,
                         orthonormal=orthonormal,
                         radius=radius)
        
        if reducer is None:
            raise ValueError("HybridSMDS requires a reducer object (e.g. PLSRegression).")

        self.reducer = reducer

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

        # --- LOGIK FÜR ISSUE #53 / #65 ---
        # Prüfen, ob y bereits die Zielkoordinaten sind
        is_direct_embedding = (y.ndim == 2) and (y.shape[1] == self.n_components)

        if is_direct_embedding:
            # Fall B: Direkte Eingabe der Koordinaten (Shape)
            # Wir überspringen _compute_ideal_distances und _classical_mds
            Y = y
            
            # Validierung: Dimensionen müssen stimmen
            if Y.shape[0] != X.shape[0]:
                raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, but target Y has {Y.shape[0]}.")
                
            self.Y_ = Y  # Speichere das direkt übergebene Target
            
        else:
            # Fall A: Standard Supervised MDS Flow (Labels -> Distanzen -> MDS -> Y)
            y = y.squeeze()
            
            # Berechne ideale Distanzen
            distances = self._compute_ideal_distances(y)

            # Check auf negative Distanzen
            if isinstance(distances, np.ndarray) and np.any(distances < 0):
                raise ValueError("HybridSMDS: does not support incomplete distance matrices.")

            # Erzeuge Embedding via MDS
            Y = self._classical_mds(distances)
            self.Y_ = Y

        # --- ENDE LOGIK #53 ---

        # 3. Trainiere den Reducer: X -> Y
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