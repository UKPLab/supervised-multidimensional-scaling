from abc import ABC, abcstractmethod

class BaseShape(ABC):
    """
    General abstraction for shapes (manifolds).
    """
    @abstractmethod
    def _compute_distance_matrix(self, y: np.ndarray) -> np.ndarray:
        """
        Compute ideal pairwise distance matrix D based on labels y for the specific shape.
        """
        raise NotImplementedError()

    def _validate_input(self):
        # TODO: Do we want to check if the input satisfies some conditions?
        pass