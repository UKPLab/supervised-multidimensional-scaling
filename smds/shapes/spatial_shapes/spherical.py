from typing import Optional

import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class SphericalShape(BaseShape):
    """
    Spherical shape for computing ideal distances on a spherical manifold (straight-line).

    Maps each coordinate into a 3D point on a sphere of radius r.
    """

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, radius: Optional[float] = 1.0, normalize_labels: bool = False):
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
        lat = np.radians(y[:, 0])
        lon = np.radians(y[:, 1])

        coords = np.stack(
            [
                self.radius * np.cos(lat) * np.cos(lon),  # x
                self.radius * np.cos(lat) * np.sin(lon),  # y
                self.radius * np.sin(lat),  # z
            ],
            axis=1,
        )

        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance: NDArray[np.float64] = np.linalg.norm(diffs, axis=2)
        return distance
