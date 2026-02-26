# Defining a Custom Shape

This guide demonstrates how to extend `smds` by defining your own custom manifold shape.
We will implement a **Circular** shape as an example.

## Purpose of Custom shapes

Custom shapes allow you to define manifold geometries that are not provided in the default smds package. This is useful when your data lies on a specific manifold structure that you want to model explicitly, so that our library does not limit your ideas!

For a list of all available built-in shapes, please refer to the [Available Shapes](../shapes.md).

## Example: custom Circle

In this example, we define a shape representing a circle.
Points are defined by a 1D scalar (angle or normalized position). The distance is the chord length through the circle.

### Implementation

```python
import numpy as np
from numpy.typing import NDArray
from smds.shapes import BaseShape

class MyCircle(BaseShape):
    y_ndim = 1

    @property
    def normalize_labels(self) -> bool:
        return self._normalize_labels

    def __init__(self, radious: float = 1.0, normalize_labels: bool = True):
        self.radious = radious
        self._normalize_labels = normalize_labels

    def _compute_distances(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = np.abs(y[:, None] - y[None, :])
        delta = np.minimum(delta, 1 - delta)
        return 2 * np.sin(np.pi * delta)
```

### Usage

```python
# Define points on the circle (normalized 0-1)
points = np.array([0.0, 0.25, 0.5])

# Instantiate the shape
circle_shape = MyCircle()

# Compute the distance matrix
distances = circle_shape(points)

print(distances)
```
