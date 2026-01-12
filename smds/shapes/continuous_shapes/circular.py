import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape


class CircularShape(BaseShape):
    """
    Compute Euclidean (chord) distances for continuous data on a circular manifold.

    This shape wraps continuous normalized values onto a circle and calculates
    the straight-line (chord) distance between them through the circle's interior.
    This differs from the arc length (geodesic) distance.

    Parameters
    ----------
    radious : float, optional
        The radius of the circle. Default is 1.0.
        (Note: The current implementation calculates distances for a unit circle
        regardless of this parameter).
    normalize_labels : bool, optional
        Whether to normalize labels to the range [0, 1]. Default is True.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (1). Expects scalar values.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, radious: float = 1.0, normalize_labels: bool = True):
        self.radious = radious
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute pairwise chord distances.

        The labels are treated as fractions of a circle (0 to 1).
        The distance is calculated as:
        $$ D_{ij} = 2 \\sin(\\pi \\cdot \\delta_{ij}) $$
        where $\\delta_{ij} = \\min(|y_i - y_j|, 1 - |y_i - y_j|)$ is the shortest
        arc fraction between points.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input 1D array of labels (normalized).

        Returns
        -------
        NDArray[np.float64]
            Pairwise Euclidean distance matrix (chord lengths).
        """
        delta: NDArray[np.float64] = np.abs(y[:, None] - y[None, :])
        delta = np.minimum(delta, 1 - delta)

        distance: NDArray[np.float64] = 2 * np.sin(np.pi * delta)
        return distance
