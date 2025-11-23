from abc import ABC, abstractmethod


class BaseCoordinates(ABC):
    @abstractmethod
    def to_cartesian(self) -> "CartesianCoordinates":  # type: ignore
        pass
