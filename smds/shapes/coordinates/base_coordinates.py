from abc import ABC, abstractmethod


class BaseCoordinates(ABC):
    """
    Abstract base class for coordinate system transformations.
    """

    @abstractmethod
    def to_cartesian(self) -> "CartesianCoordinates":  # type: ignore
        """
        Convert the stored coordinates to Cartesian space.

        Returns
        -------
        CartesianCoordinates
            An instance representing the equivalent points in Cartesian (x, y, z...)
            coordinates.
        """
        pass
