from base_stress import BaseStress
from numpy.typing import NDArray


class ShepardGoodnessScore(BaseStress):
    def compute(self, X_high: NDArray, X_low: NDArray) -> float:
        raise NotImplementedError
