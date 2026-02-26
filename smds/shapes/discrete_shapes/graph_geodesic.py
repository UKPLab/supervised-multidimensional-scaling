import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path  # type: ignore[import-untyped]
from sklearn.neighbors import kneighbors_graph  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape


class GraphGeodesicShape(BaseShape):
    """
    A generic discrete shape hypothesis that approximates the manifold structure
    using a K-Nearest Neighbors (KNN) Geodesic Graph (Isomap approach).

    Logic:
    1. Graph Construction: Connects each point to its 'k' nearest neighbors.
    2. Geodesic Distance: Defines distance between two points as the shortest
       path along the graph edges, rather than the straight Euclidean line.

    This allows the model to "unfold" arbitrary curved shapes (like Swiss Rolls
    or folded paper) without knowing their equation beforehand.

    Hyperparameters:
        n_neighbors (int): Number of neighbors to connect. Default is 5.
                           - Too small: Graph might be disconnected.
                           - Too large: Short-circuits the manifold (shortcuts).
    """

    def __init__(self, n_neighbors: int = 5, normalize_labels: bool = True):
        self.n_neighbors = n_neighbors
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        """Dimensionality of the input labels."""
        return 3

    @property
    def normalize_labels(self) -> bool:
        """Whether to normalize the input labels."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc = np.asarray(y, dtype=np.float64)
        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)
        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        y_proc = np.asarray(y, dtype=np.float64)
        n_samples = y_proc.shape[0]

        effective_k = min(self.n_neighbors, n_samples - 1)
        if effective_k < 1:
            effective_k = 1

        graph = kneighbors_graph(y_proc, n_neighbors=effective_k, mode="distance", include_self=False)

        dist_matrix = shortest_path(csgraph=graph, method="auto", directed=False)

        result = np.asarray(dist_matrix, dtype=np.float64)
        return result
