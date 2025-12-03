import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class CircularShape(BaseShape):
    """
    Circular shape for computing ideal distances on a circular manifold.

    Transforms continuous values into pairwise distances assuming they lie
    on a circle, where the distance wraps around (e.g., 0.9 and 0.1 are close).
    """

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, radious: float = 1.0, normalize_labels: bool = True):
        self.radious = radious
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        delta: NDArray[np.float64] = np.abs(y[:, None] - y[None, :])
        delta = np.minimum(delta, 1 - delta)

        distance: NDArray[np.float64] = 2 * np.sin(np.pi * delta)
        return distance
