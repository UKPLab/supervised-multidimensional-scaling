import numpy as np
from numpy.typing import NDArray

from smds.shapes import BaseShape
from smds.shapes.coordinates import PolarCoordinates


class SpiralShape(BaseShape):
    """
    Arrange points in an Archimedean spiral pattern.

    This class generates a shape where points are arranged along a spiral
    trajectory defined by the equation $r = a + b\theta$. The distance metric
    is computed based on the Euclidean distance between these points in
    Cartesian coordinates.

    Parameters
    ----------
    initial_radius : float, optional
        The starting radius of the spiral (offset from the origin), corresponding
        to $a$ in the Archimedean spiral equation. Default is 0.5.
    growth_rate : float, optional
        The rate at which the spiral expands away from the center for every radian
        of rotation, corresponding to $b$ in the Archimedean spiral equation.
        Default is 1.0.
    num_turns : float, optional
        The total number of complete rotations the spiral makes. This scales the
        input labels mapping them to the angle $\theta$. Default is 2.0.
    normalize_labels : bool, optional
        Whether to normalize the input labels `y` to the range [0, 1] before
        computing the spiral coordinates. Default is True.

    Attributes
    ----------
    y_ndim : int
        The dimensionality of the label array expected by this shape (1).
    initial_radius : float
        The configured starting radius.
    growth_rate : float
        The configured growth rate.
    num_turns : float
        The configured number of turns.
    """

    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        """
        bool: Whether input labels are normalized to [0, 1].
        """
        return self._normalize_labels

    def __init__(
        self,
        initial_radius: float = 0.5,
        growth_rate: float = 1.0,
        num_turns: float = 2.0,
        normalize_labels: bool = True,
    ) -> None:
        self.initial_radius = initial_radius
        self.growth_rate = growth_rate
        self.num_turns = num_turns
        self._normalize_labels = normalize_labels

    @staticmethod
    def _do_normalize_labels(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize an array of labels to the range [0, 1].

        If the range of `y` (peak-to-peak) is 0, an array of zeros is returned
        to avoid division by zero.

        Parameters
        ----------
        y : NDArray[np.float64]
            Input array of labels to be normalized.

        Returns
        -------
        NDArray[np.float64]
            The normalized array with values scaled between 0 and 1.
        """
        y_range = np.ptp(y)
        if y_range == 0:
            zero_array: NDArray[np.float64] = np.zeros_like(y)
            return zero_array
        result: NDArray[np.float64] = (y - y.min()) / y_range
        return result

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the pairwise distance matrix for points mapped to the spiral.

        The input labels `y` are first converted to polar coordinates $(r, \theta)$
        using the class parameters:
        $$ \theta = y \times 2\\pi \times \text{num\\_turns} $$
        $$ r = \text{initial\\_radius} + \text{growth\\_rate} \times \theta $$

        These polar coordinates are converted to Cartesian coordinates, and the
        Euclidean distances between all pairs of points are calculated.

        Parameters
        ----------
        y : NDArray[np.float64]
            The input labels/values to map onto the spiral.

        Returns
        -------
        NDArray[np.float64]
            A 2D array representing the pairwise Euclidean distances between
            the transformed points.
        """
        theta = y * 2 * np.pi * self.num_turns
        radius = self.initial_radius + self.growth_rate * theta

        polar = PolarCoordinates(radius, theta)
        cartesian = polar.to_cartesian()

        result: NDArray[np.float64] = cartesian.compute_distances()
        return result
