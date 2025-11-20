import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class CylindricalShape(BaseShape):
    """
    Cylindrical shape for computing ideal distances on a cylindrical manifold (straight-line).

    Maps latitude to vertical height and longitude to angle around a cylinder of radius r.
    """

    def __init__(self, radius: float = 1.0):
        self.radius = radius

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
        # todo: maybe normalize latitiude to radius?
        lat = np.radians(y[:, 0])  # latitude as height
        lon = np.radians(y[:, 1])  # longitude as angle

        coords = np.stack(
            [
                self.radius * np.cos(lon),
                self.radius * np.sin(lon),
                lat,  # treat lat as height
            ],
            axis=1,
        )

        diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance = np.linalg.norm(diffs, axis=2)
        return distance
