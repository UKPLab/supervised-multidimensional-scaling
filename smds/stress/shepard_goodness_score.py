import numpy as np
from numpy.typing import NDArray

from smds.stress.base_stress import BaseStress


class ShepardGoodnessScore(BaseStress):
    def compute(self, D_high: NDArray[np.float64], D_low: NDArray[np.float64]) -> float:
        raise NotImplementedError
