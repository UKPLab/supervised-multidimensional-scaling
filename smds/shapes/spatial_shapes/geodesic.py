import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class GeodesicShape(BaseShape):
    """
    Compute geodesic distances on a spherical manifold (great-circle distance https://en.wikipedia.org/wiki/Great-circle_distance).

    Parameters
    ----------
    radius : float, optional
        Radius of the sphere. Default is 1.0.
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is False.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (2). Expects (n_samples, 2) for lat/lon.
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
            Input coordinates (latitude, longitude) in degrees.

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
        Calculate the great-circle distance between points using the Haversine formula.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples, 2) containing latitude and longitude
            in degrees.

        Returns
        -------
        NDArray[np.float64]
            Pairwise distance matrix.
        """
        lat = np.radians(y[:, 0])[:, np.newaxis]
        lon = np.radians(y[:, 1])[:, np.newaxis]

        dlat = lat - lat.T
        dlon = lon - lon.T

        lat1 = lat
        lat2 = lat.T

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        a = np.clip(a, 0, 1)  # prevent floating-point error
        c = 2 * np.arcsin(np.sqrt(a))
        distance: NDArray[np.float64] = self.radius * c
        return distance
