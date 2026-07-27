import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import shortest_path  # type: ignore[import-untyped]
from sklearn.neighbors import kneighbors_graph  # type: ignore[import-untyped]

from smds.shapes.base_shape import BaseShape


class GraphGeodesicShape(BaseShape):
    """
    Approximate manifold distances via a k-nearest-neighbors geodesic graph.

    Constructs a KNN graph over the input coordinates and computes the shortest
    path between all pairs (Isomap approach). This lets the model respect the
    intrinsic geometry of curved manifolds without knowing their equation.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors to connect each point to. Default is 5.
        Too small a value may leave the graph disconnected; too large a value
        may introduce shortcuts that distort the geodesic distances.
    normalize_labels : bool, optional
        Whether to normalize labels using the base class logic. Default is True.

    Attributes
    ----------
    y_ndim : int
        Dimensionality of input labels (2). Expects (n_samples, n_features).
    """

    def __init__(self, n_neighbors: int = 5, normalize_labels: bool = True):
        self.n_neighbors = n_neighbors
        self._normalize_labels_flag = normalize_labels

    @property
    def y_ndim(self) -> int:
        """int: Dimensionality of the input labels (2)."""
        return 2

    @property
    def normalize_labels(self) -> bool:
        """bool: Whether to normalize the input labels."""
        return self._normalize_labels_flag

    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate and coerce input to a 2D float64 array.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        NDArray[np.float64]
            Validated 2D array of shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If ``y`` has more than 2 dimensions after reshaping.
        """
        y_proc = np.asarray(y, dtype=np.float64)
        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)
        if y_proc.ndim != 2:
            raise ValueError(f"GraphGeodesicShape expects 2D input (n_samples, n_features), got {y_proc.ndim}D.")
        return y_proc

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Build a KNN graph and return all-pairs shortest-path distances.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.float64]
            Pairwise geodesic distance matrix of shape (n_samples, n_samples).
        """
        y_proc = np.asarray(y, dtype=np.float64)
        n_samples = y_proc.shape[0]

        effective_k = min(self.n_neighbors, n_samples - 1)
        if effective_k < 1:
            effective_k = 1

        graph = kneighbors_graph(y_proc, n_neighbors=effective_k, mode="distance", include_self=False)

        dist_matrix = shortest_path(csgraph=graph, method="auto", directed=False)

        result: NDArray[np.float64] = np.asarray(dist_matrix, dtype=np.float64)
        return result
