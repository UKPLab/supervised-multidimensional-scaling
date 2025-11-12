import numpy as np
from numpy.typing import NDArray
from smds.shapes.base_shape import BaseShape
from scipy.spatial.distance import cdist


class CircularShape(BaseShape):
    """
    Circular shape for computing ideal distances on a circular manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a circle, where the distance wraps around (e.g., 0.9 and 0.1 are close).
    """
    def __init__(self, radious: float = 1.0):
        self.radious = radious

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm: NDArray[np.float64] = self._normalize_y(y)
        
        delta: NDArray[np.float64] = np.abs(y_norm[:, None] - y_norm[None, :])
        delta = np.minimum(delta, 1 - delta)
        
        distance: NDArray[np.float64] = 2 * np.sin(np.pi * delta)
        return distance

class LinearShape(BaseShape):
    """
    Linear shape for computing ideal distances on a linear manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a line, where the distance is the linear difference.
    """
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        diff: NDArray[np.float64] = y[:, None] - y[None, :]
        distance: NDArray[np.float64] = np.abs(diff)
        return distance
    

class EuclideanShape(LinearShape):
    """
    Euclidean shape for computing ideal distances on a Euclidean manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a Euclidean space, where the distance is the Euclidean distance.
    """
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        diff: NDArray[np.float64] = y[:, None] - y[None, :]
        distance: NDArray[np.float64] = np.abs(diff)
        return distance

class LogLinearShape(LinearShape):
    """
    Log-linear shape for computing ideal distances on a log-linear manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a log-linear space, where the distance is the log-linear difference.
    """

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc: NDArray[np.float64] = super()._validate_input(y)
        if np.any(y_proc <= 0):
            raise ValueError("Input values must be positive.")
        return y_proc
        
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        log_y: NDArray[np.float64] = np.log(y + 1)
        distance: NDArray[np.float64] = np.abs(log_y[:, None] - log_y[None, :])
        return distance

class HelixShape(BaseShape):
    """
    Helix shape for computing ideal distances on a helix manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a helix, where the distance is the helix difference.
    """
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm: NDArray[np.float64] = self._normalize_y(y)
        
        theta: NDArray[np.float64] = 2 * np.pi * y_norm
        spiral_coords: NDArray[np.float64] = np.column_stack([
            np.cos(theta),
            np.sin(theta),
            y_norm
        ])
        
        distance: NDArray[np.float64] = cdist(spiral_coords, spiral_coords, metric='euclidean')
        return distance
    
class SemicircularShape(BaseShape):
    """
    Semicircular shape for computing ideal distances on a semicircular manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a semicircular space, where the distance is the semicircular difference.
    """
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm: NDArray[np.float64] = self._normalize_y(y)
        
        delta: NDArray[np.float64] = np.abs(y_norm[:, None] - y_norm[None, :]) 
        distance: NDArray[np.float64] = 2 * np.sin((np.pi / 2) * delta)
        return distance

class LogSemicircularShape(BaseShape):
    """
    Log-semicircular shape for computing ideal distances on a log-semicircular manifold.
    
    Transforms continuous values into pairwise distances assuming they lie
    on a log-semicircular space, where the distance is the log-semicircular difference.
    """
    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc: NDArray[np.float64] = super()._validate_input(y)
        if np.any(y_proc <= 0):
            raise ValueError("Input values must be positive.")
        return y_proc
    
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_norm: NDArray[np.float64] = self._normalize_y(y)
        
        y_log: NDArray[np.float64] = np.log(y_norm + 1)
        delta: NDArray[np.float64] = np.abs(y_log[:, None] - y_log[None, :])
        distance: NDArray[np.float64] = 2 * np.sin((np.pi / 2) * delta)
        return distance