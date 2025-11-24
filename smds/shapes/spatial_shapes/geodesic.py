from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class GeodesicShape(BaseShape):
    """
    Geodesic shape for computing ideal distances on a spherical manifold (great-circle).

    Straight-line distances between points in Euclidean space that lie on a sphere.
    """

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, radius: Optional[float] = 1.0, normalize_labels: Optional[bool] = False):
        self.radius = radius
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 2 or y_proc.shape[1] != 2:
            raise ValueError(
                f"Input 'y' must be 2-dimensional (n_samples, 2), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )

        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        lat = np.radians(y[:, 0])[:, np.newaxis]
        lon = np.radians(y[:, 1])[:, np.newaxis]

        dlat = lat - lat.T
        dlon = lon - lon.T

        lat1 = lat
        lat2 = lat.T

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        a = np.clip(a, 0, 1)  # prevent floating-point error
        c = 2 * np.arcsin(np.sqrt(a))
        distance = self.radius * c
        return distance
