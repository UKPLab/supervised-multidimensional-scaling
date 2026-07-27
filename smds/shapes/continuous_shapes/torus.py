import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape


class TorusShape(BaseShape):
    """
    Compute geodesic distances on a flat torus (T\\ :sup:`1` × T\\ :sup:`1`) manifold.

    Models data lying on the product of two circles. Input coordinates are
    mapped onto the unit square [0, 1] × [0, 1] with periodic boundary
    conditions in both directions:

    - (u, 0) ~ (u, 1) — cylinder wrap in the v direction.
    - (0, v) ~ (1, v) — cylinder wrap in the u direction.

    If the input has more than 2 dimensions it is first reduced to 2 via PCA;
    if it has fewer than 2 dimensions zeros are padded.

    Parameters
    ----------
    radii : tuple of (float, float), optional
        Scaling weights (r1, r2) for the two cyclic dimensions.
        Default is ``(1.0, 1.0)`` (flat Clifford torus).
    normalize_labels : bool, optional
        Whether to normalize labels to the range [0, 1]. Default is True.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (2). Expects (u, v) parameters.
    """

    def __init__(self, radii: tuple[float, float] = (1.0, 1.0), normalize_labels: bool = True):
        self._radii = radii
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        """int: Expected input dimensionality (2)."""
        return 2

    @property
    def radii(self) -> tuple[float, float]:
        """Tuple of (float, float): Scaling weights for the two cyclic dimensions."""
        return self._radii

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether labels are normalized to the unit square."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input and reduce or pad to exactly 2 dimensions.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        NDArray[np.float64]
            Array of shape (n_samples, 2) suitable for torus projection.

        Raises
        ------
        ValueError
            If ``y`` is empty.
        """
        y_proc = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)

        n_samples, n_features = y_proc.shape

        if n_features > 2:
            n_components = 2
            if n_samples < n_components:
                y_new = np.zeros((n_samples, 2))
                y_new[:, : min(n_features, 2)] = y_proc[:, : min(n_features, 2)]
                y_proc = y_new
            else:
                pca = PCA(n_components=2)
                y_proc = pca.fit_transform(y_proc)
        elif n_features < 2:
            zeros = np.zeros((n_samples, 2 - n_features))
            y_proc = np.hstack([y_proc, zeros])

        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute geodesic distances on the flat torus.

        Calculates the shortest-path distance respecting periodic boundaries
        in both u and v directions:

        .. math::

            D_{ij} = \\sqrt{(r_1 \\cdot \\delta u_{ij})^2 + (r_2 \\cdot \\delta v_{ij})^2}

        where :math:`\\delta u = \\min(|u_i - u_j|,\\, 1 - |u_i - u_j|)` and
        analogously for :math:`\\delta v`.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples, 2), assumed normalized to [0, 1].

        Returns
        -------
        NDArray[np.float64]
            Pairwise geodesic distance matrix of shape (n_samples, n_samples).
        """
        u = y[:, 0]
        v = y[:, 1]

        r1, r2 = self._radii

        u1 = u.reshape(-1, 1)
        u2 = u.reshape(1, -1)
        v1 = v.reshape(-1, 1)
        v2 = v.reshape(1, -1)

        diff_u = np.abs(u1 - u2)
        diff_v = np.abs(v1 - v2)

        circ_diff_u = np.minimum(diff_u, 1.0 - diff_u)
        circ_diff_v = np.minimum(diff_v, 1.0 - diff_v)

        dist_sq = (r1 * circ_diff_u) ** 2 + (r2 * circ_diff_v) ** 2

        result: NDArray[np.float64] = np.sqrt(dist_sq)
        return result
