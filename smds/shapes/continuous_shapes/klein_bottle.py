import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class KleinBottleShape(BaseShape):
    """
    Manifold hypothesis representing a Klein Bottle topology.

    This shape assumes the data lies on a 2D surface that is non-orientable.

    Logic:
    1. Dimensionality: Requires exactly 2 dimensions as (u, v) parameters.
       If < 2 dimensions, zeros are padded. If > 2 dimensions, raises an error.
    2. Parametrization: Maps these 2 dimensions to the unit square [0, 1] x [0, 1].
    3. Distance Calculation: Computes pairwise distances respecting the Klein
       bottle identifications:
       - Top/Bottom edges match (Cylinder): (u, 0) ~ (u, 1)
       - Left/Right edges match with a Twist (Möbius): (0, v) ~ (1, 1-v)
    """

    y_ndim = 2

    def __init__(self, normalize_labels: bool = True):
        self._normalize_labels_flag = normalize_labels

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether input labels are normalized."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate input is 2D with shape (n_samples, 2).

        Parameters
        ----------
        y : NDArray[np.float64]
            Input coordinates (u, v) parameters for Klein Bottle.

        Returns
        -------
        NDArray[np.float64]
            Validated input array.

        Raises
        ------
        ValueError
            If input is empty or shape is not (n_samples, 2).
        """
        y_proc = np.asarray(y, dtype=np.float64)

        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")

        if y_proc.ndim != 2 or y_proc.shape[1] != 2:
            raise ValueError(
                f"Klein Bottle requires exactly 2 dimensions (u, v). "
                f"Expected shape (n_samples, 2), but got shape {y_proc.shape}."
            )

        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes geodesic distances on a Flat Klein Bottle.
        Assumes y is normalized to [0, 1].
        """
        u = y[:, 0]  # Twist axis
        v = y[:, 1]  # Cylinder axis

        u1 = u.reshape(-1, 1)
        u2 = u.reshape(1, -1)
        v1 = v.reshape(-1, 1)
        v2 = v.reshape(1, -1)

        diff_u = np.abs(u1 - u2)
        diff_v = np.abs(v1 - v2)

        # 1. Direct
        dist_sq_direct = diff_u**2 + diff_v**2

        # 2. Cylinder Wrap (Wrap V)
        dist_sq_cylinder = diff_u**2 + (1.0 - diff_v) ** 2

        # 3. Möbius Twist (Wrap U -> Flip V)
        dist_u_twist = 1.0 - diff_u
        dist_v_flipped = np.abs(v1 + v2 - 1.0)

        dist_sq_twist = dist_u_twist**2 + dist_v_flipped**2

        # 4. Combined Wrap (Wrap U + Wrap V)
        dist_v_flipped_wrapped = np.minimum(dist_v_flipped, 1.0 - dist_v_flipped)
        dist_sq_twist_wrap = dist_u_twist**2 + dist_v_flipped_wrapped**2

        # Minimum distance
        D_sq = np.minimum(dist_sq_direct, dist_sq_cylinder)
        D_sq = np.minimum(D_sq, dist_sq_twist)
        D_sq = np.minimum(D_sq, dist_sq_twist_wrap)

        result: NDArray[np.float64] = np.sqrt(D_sq)
        return result
