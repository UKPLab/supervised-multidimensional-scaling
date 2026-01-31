import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape


class KleinBottleShape(BaseShape):
    r"""
    Manifold hypothesis representing a Klein Bottle topology.

    This shape assumes the data lies on a 2D surface that is non-orientable.
    It treats the feature space as a "Flat Klein Bottle," which is formed by
    identifying the edges of a unit square:
    1.  **Cylinder:** Top and Bottom edges match $(u, 0) \sim (u, 1)$.
    2.  **Möbius:** Left and Right edges match with a twist $(0, v) \sim (1, 1-v)$.

    Parameters
    ----------
    normalize_labels : bool, optional
        Whether to normalize labels to the unit square [0, 1]. Default is True.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of the manifold surface (2).
    """

    def __init__(self, normalize_labels: bool = True):
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        """int: The expected dimensionality of the reduced feature space (2)."""
        return 2

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input and project it to the required 2D surface.

        If the input has more than 2 dimensions, PCA is used to project it down
        to the 2 most significant components (capturing maximum variance).
        If it has fewer than 2 dimensions, it is padded with zeros.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input data of arbitrary dimensionality.

        Returns
        -------
        NDArray[np.float64]
            A (n_samples, 2) array ready for Klein bottle distance computation.
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
                # Fallback if samples < components (PCA requires n_samples >= n_components)
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
        Compute geodesic distances on a Flat Klein Bottle.

        Calculates the minimum distance between points considering the four possible
        paths on the fundamental polygon:
        1.  **Direct:** Standard Euclidean distance.
        2.  **Cylinder Wrap:** Crossing top/bottom boundaries (wrapping $v$).
        3.  **Twist (Möbius):** Crossing left/right boundaries (wrapping $u$),
            which flips the $v$ coordinate ($v \to 1-v$).
        4.  **Twist + Wrap:** Crossing both boundaries.

        Parameters
        ----------
        y : NDArray[np.float64]
            The projected and normalized 2D coordinates in [0, 1].

        Returns
        -------
        NDArray[np.float64]
            Pairwise geodesic distance matrix.
        """
        u = y[:, 0]  # Twist axis
        v = y[:, 1]  # Cylinder axis

        u1 = u.reshape(-1, 1)
        u2 = u.reshape(1, -1)
        v1 = v.reshape(-1, 1)
        v2 = v.reshape(1, -1)

        diff_u = np.abs(u1 - u2)
        diff_v = np.abs(v1 - v2)

        # 1. Direct path
        dist_sq_direct = diff_u**2 + diff_v**2

        # 2. Cylinder Wrap (Wrap V: |v1 - v2| becomes 1 - |v1 - v2|)
        dist_sq_cylinder = diff_u**2 + (1.0 - diff_v) ** 2

        # 3. Möbius Twist (Wrap U -> Flip V)
        dist_u_twist = 1.0 - diff_u
        dist_v_twist = np.abs(v1 + v2 - 1.0)  # |v1 - (1 - v2)| = |v1 + v2 - 1|

        dist_sq_twist = dist_u_twist**2 + dist_v_twist**2

        # 4. Combined Wrap (Wrap U + Wrap V)
        dist_v_twist_wrap = 1.0 - dist_v_twist
        dist_sq_twist_wrap = dist_u_twist**2 + dist_v_twist_wrap**2

        # Minimum distance
        D_sq = np.minimum(dist_sq_direct, dist_sq_cylinder)
        D_sq = np.minimum(D_sq, dist_sq_twist)
        D_sq = np.minimum(D_sq, dist_sq_twist_wrap)

        result: NDArray[np.float64] = np.sqrt(D_sq)
        return result
