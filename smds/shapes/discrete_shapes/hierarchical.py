import numpy as np
from numpy.typing import NDArray
from typing import List

from smds.shapes.base_shape import BaseShape


class HierarchicalShape(BaseShape):
    """
    Implements hierarchical shape for multi-level categorical labels.

    The distance between two points is determined by the first (most significant)
    hierarchical level at which they differ. Each level has an associated distance value.
    """
    
    def __init__(self, level_distances: List[float]) -> None:
        """
        Initialize HierarchicalShape with level distances.

        Args:
            level_distances: List of distance values for each hierarchical level.
                            The length determines the number of levels expected in y.
        """
        if not level_distances:
            raise ValueError("level_distances cannot be empty.")
        if any(d < 0 for d in level_distances):
            raise ValueError("All level_distances must be non-negative.")
        self.level_distances = np.array(level_distances, dtype=np.float64)
    
    def _validate_input(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Validate that y is a 2D array with number of columns matching level_distances length.

        Args:
            y: Input array of hierarchical labels.

        Returns:
            Validated and processed array.
        """
        y_proc: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        
        if y_proc.size == 0:
            raise ValueError("Input 'y' cannot be empty.")
        
        if y_proc.ndim != 2:
            raise ValueError(
                f"Input 'y' must be 2-dimensional (n_samples, n_levels), "
                f"but got shape {y_proc.shape} with {y_proc.ndim} dimensions."
            )
        
        expected_cols = len(self.level_distances)
        if y_proc.shape[1] != expected_cols:
            raise ValueError(
                f"Input 'y' must have {expected_cols} columns (matching level_distances length), "
                f"but got {y_proc.shape[1]} columns."
            )
        
        return y_proc
    
    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute pairwise distances based on hierarchical levels.

        Args:
            y: A 2D numpy array of shape (n_samples, n_levels) containing
               hierarchical categorical labels.

        Returns:
            A (n_samples, n_samples) distance matrix.
        """
        differences = y[:, None, :] != y[None, :, :]
        
        first_diff_level = np.argmax(differences, axis=2)
        has_difference = np.any(differences, axis=2)
        
        distance_matrix = np.where(
            has_difference,
            self.level_distances[first_diff_level],
            0.0
        )
        
        return distance_matrix

