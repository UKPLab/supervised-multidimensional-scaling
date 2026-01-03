import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape


class TorusShape(BaseShape):
    """
    Manifold hypothesis representing a Flat Torus topology.
    This shape assumes the data lies on the product of two circles (T1 x T1).

    Logic:
    1. Dimensionality Reduction: Projects input 'y' to 2 dimensions via PCA
       (or padding) to capture the two independent cycles.
    2. Parametrization: Maps these 2 dimensions to the unit square [0, 1] x [0, 1].
    3. Distance Calculation: Computes pairwise distances respecting the Torus
       identifications (periodic boundary conditions in both directions):
       - Top/Bottom edges match: (u, 0) ~ (u, 1)
       - Left/Right edges match: (0, v) ~ (1, v)

    Hyperparameters:
        radii: A tuple (r1, r2) weighting the two cyclic dimensions.
               Default is (1.0, 1.0) (Flat Clifford Torus).
    """

    def __init__(self, radii: tuple[float, float] = (1.0, 1.0), normalize_labels: bool = True):
        self._radii = radii
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        return 2

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validates input and reduces dimensions if necessary via PCA.
        Matches KleinBottleShape logic.
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
        Computes geodesic distances on a Flat Torus.
        Assumes y is normalized to [0, 1].
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