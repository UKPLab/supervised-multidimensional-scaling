# Available shapes

This page lists all the built-in manifold shapes available in `smds`. For shapes not listed here check out how to add your  [custom shape](./examples/custom_shape.md).

## Continuous Shapes
Shapes for continuous data manifolds.

- `CircularShape`: For data on a circle (1D).
- `EuclideanShape`: For linear data (1D).
- `KleinBottleShape`: For data on a Klein bottle surface.
- `LogLinearShape`: For data with logarithmic scaling.
- `SemicircularShape`: For data on a semicircle.
- `SpiralShape`: For data on a spiral.

## Discrete Shapes
Shapes for discrete or categorical data structures.

- `ChainShape`: For linear sequences of discrete nodes.
- `ClusterShape`: For clustered data.
- `DiscreteCircularShape`: For discrete points on a circle (cycle graph).
- `HierarchicalShape`: For hierarchical/tree-structured data.

## Spatial shapes
Shapes for data in 3D space or on surfaces.

- `CylindricalShape`: For data on a cylinder.
- `GeodesicShape`: For data on a manifold using geodesic distances.
- `SphericalShape`: For data on a sphere.

## Base Classs
All shapes inherit from the base class:

- `BaseShape`: The abstract base class for all shapes.
