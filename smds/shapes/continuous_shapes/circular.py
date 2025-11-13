import numpy as np
from numpy.typing import NDArray
from smds.shapes.base_shape import BaseShape


class CircularShape(BaseShape):
    """
    Circular shape for computing ideal distances on a circular manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a circle, where the distance wraps around (e.g., 0.9 and 0.1 are close).
    """
    def __init__(self, radious: float = 1.0):
        self.radious = radious

    def _normalize_y(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize y to [0, 1] range with edge case handling.
        
        If all values are identical (max_y == min_y), returns zeros array.
        This ensures that subsequent distance calculations result in zero distances.
        
        Returns:
            y_norm: Normalized array in [0, 1] range, or zeros if all values are identical
        """
        max_y: np.float64 = np.max(y)
        min_y: np.float64 = np.min(y)

        if max_y == min_y:
            return np.zeros_like(y, dtype=float)

        y_norm: NDArray[np.float64] = (y - min_y) / (max_y - min_y)
        return y_norm

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm: NDArray[np.float64] = self._normalize_y(y)

        delta: NDArray[np.float64] = np.abs(y_norm[:, None] - y_norm[None, :])
        delta = np.minimum(delta, 1 - delta)

        distance: NDArray[np.float64] = 2 * np.sin(np.pi * delta)
        return distance
