import numpy as np
from numpy.typing import NDArray

from smds.shapes.base_shape import BaseShape


class PolytopeShape(BaseShape):
    """
    Manifold hypothesis representing a Polytope topology.

    This shape logic tries to push the n_clusters passed as an argument
    as far away as possible from each other on an n-dimensional unit sphere
    using iterative repulsion.
    """

    def __init__(
        self,
        n_dim: int = 3,
        n_iter: int = 500,
        lr: float = 0.1,
        seed: int | None = None,
        normalize_labels: bool = False,
    ):
        self.n_dim = n_dim
        self.n_iter = n_iter
        self.lr = lr
        self.seed = seed
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        """Dimensionality of the input labels."""
        return 1

    @property
    def normalize_labels(self) -> bool:
        """Whether to normalize the input labels."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc = np.asarray(y, dtype=np.float64).squeeze()
        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")
        if y_proc.ndim == 0:
            y_proc = y_proc.reshape(1)
        elif y_proc.ndim > 1:
            raise ValueError("PolytopeShape expects 1D cluster labels.")
        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generates a distance matrix D between all points in y, where points with
        different labels are maximally spread out in n_dim dimensions.
        """
        rng = np.random.default_rng(self.seed)
        labels, inverse_indices = np.unique(y, return_inverse=True)
        C = len(labels)

        V = rng.normal(size=(self.n_dim, C))
        V /= np.linalg.norm(V, axis=0, keepdims=True)

        for _ in range(self.n_iter):
            diff = V[:, :, None] - V[:, None, :]

            dist_sq = np.sum(diff**2, axis=0) + 1e-8
            inv_dist = 1.0 / dist_sq

            np.fill_diagonal(inv_dist, 0)

            forces = np.sum(diff * inv_dist[None, :, :], axis=2)
            V += self.lr * forces
            V /= np.linalg.norm(V, axis=0, keepdims=True)

        y_vertices = V[:, inverse_indices].T

        diff_y = y_vertices[:, None, :] - y_vertices[None, :, :]
        dist_matrix: NDArray[np.float64] = np.linalg.norm(diff_y, axis=-1)

        return dist_matrix
