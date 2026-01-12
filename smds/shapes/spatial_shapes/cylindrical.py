import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class CylindricalShape(BaseShape):
    """
    Compute Euclidean distances between points mapped onto a cylinder.

    This class maps input coordinates to a 3D cylindrical surface. One dimension
    is treated as the height (linear) and the other as the angle (circular) around
    a cylinder of fixed radius.

    Parameters
    ----------
    radius : float, optional
        The radius of the cylinder. Default is 1.0.
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (2). Expects (n_samples, 2).
    """

    y_ndim = 2

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, radius: float = 1.0, normalize_labels: bool = False):
        self.radius = radius
        self._normalize_labels = normalize_labels

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input is 2D with shape (n_samples, 2).

        Parameters
        ----------
        y : NDArray[np.float64]
            Input coordinates (e.g., latitude/height, longitude/angle).

        Returns
        -------
        NDArray[np.float64]
            Validated input array.

        Raises
        ------
        ValueError
            If input is empty or shape is not (n_samples, 2).
        """
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
        """
        Map inputs to cylindrical coordinates and compute Euclidean norms.

        Maps input (lat, lon) to $(x, y, z)$ where:
        - $x, y$ are derived from `lon` (angle) and `radius`.
        - $z$ is derived directly from `lat` (height).

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples, 2). By convention, column 0 is
            treated as the vertical component (latitude/height) and column 1
            as the angular component (longitude).

        Returns
        -------
        NDArray[np.float64]
            Pairwise Euclidean distance matrix through 3D space.
        """
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
        distance: NDArray[np.float64] = np.linalg.norm(diffs, axis=2)
        return distance
