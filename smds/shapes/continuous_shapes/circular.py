import numpy as np

from smds.shapes.base_shape import BaseShape


class CircularShape(BaseShape):
    def __init__(self, radious: float = 1.0) -> None:
        self.radious = radious

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self._compute_distance_matrix(y)

    def _compute_distance_matrix(self, y: np.ndarray) -> np.ndarray:
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        delta = np.abs(y_norm[:, None] - y_norm[None, :])
        delta = np.minimum(delta, 1 - delta)
        return 2 * np.sin(np.pi * delta)
