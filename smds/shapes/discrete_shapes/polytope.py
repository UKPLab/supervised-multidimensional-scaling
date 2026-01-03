import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph

from smds.shapes.base_shape import BaseShape


class PolytopeShape(BaseShape):
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

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Graph Geodesic distance (Shortest Path) between all points.
        """
        y_proc = np.asarray(y, dtype=np.float64)
        n_samples = y_proc.shape[0]

        effective_k = min(self.n_neighbors, n_samples - 1)
        if effective_k < 1:
            effective_k = 1

        graph = kneighbors_graph(y_proc, n_neighbors=effective_k, mode="distance", include_self=False)

        dist_matrix = shortest_path(csgraph=graph, method="auto", directed=False)

        result: NDArray[np.float64] = dist_matrix
        return result
