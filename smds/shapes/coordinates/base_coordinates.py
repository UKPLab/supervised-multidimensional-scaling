from abc import ABC, abstractmethod

from smds.shapes.coordinates.cartesian_coordinates import CartesianCoordinates


class BaseCoordinates(ABC):
    @abstractmethod
    def to_cartesian(self) -> CartesianCoordinates:
        raise NotImplementedError()
