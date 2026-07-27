import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform  # type: ignore[import-untyped]

from smds.shapes import BaseShape


class EuclideanShape(BaseShape):
    """
    Compute Euclidean (linear) distances for continuous data in n dimensions.

    This shape models data lying in a flat Euclidean space of arbitrary
    dimensionality. Inputs of shape (n_samples,) are treated as points on a
    line, inputs of shape (n_samples, n_features) as points in
    :math:`\\mathbb{R}^{n\\_features}`.

    Reference: Table 1 in "Shape Happens" paper (referred to as 'linear').

    Parameters
    ----------
    normalize_labels : bool, optional
        Whether to normalize labels to the range [0, 1]. Default is True.

    Attributes
    ----------
    y_ndim : tuple of int
        Accepted label dimensionalities, ``(1, 2)``: either scalar labels of
        shape (n_samples,) or coordinate arrays of shape
        (n_samples, n_features).

    Notes
    -----
    :class:`~smds.SupervisedMDS` defaults ``n_components`` to 1 for this shape,
    since scalar labels are the common case. With (n_samples, n_features)
    labels you almost always want to pass ``n_components=n_features``
    explicitly; otherwise the embedding is truncated to a single dimension.
    """

    y_ndim = (1, 2)

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels

    def __init__(self, normalize_labels: bool = True):
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute pairwise Euclidean distances.

        Calculates:

        .. math::

            D_{ij} = \\sqrt{\\sum_{k} (y_{ik} - y_{jk})^2}

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        NDArray[np.float64]
            Pairwise Euclidean distance matrix of shape (n_samples, n_samples).
        """
        points = y.reshape(len(y), -1)

        distance_matrix: NDArray[np.float64] = squareform(pdist(points))

        return distance_matrix
