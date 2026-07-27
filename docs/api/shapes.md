# Shapes

All shapes inherit from `BaseShape` and implement the same callable interface:
calling an instance with an array of labels returns a pairwise distance matrix.

## Base

::: smds.shapes.base_shape.BaseShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

---

## Continuous Shapes

### CircularShape

::: smds.shapes.continuous_shapes.circular.CircularShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### SemicircularShape

::: smds.shapes.continuous_shapes.semicircular.SemicircularShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### EuclideanShape

::: smds.shapes.continuous_shapes.euclidean.EuclideanShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### LogLinearShape

::: smds.shapes.continuous_shapes.log_linear.LogLinearShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### SpiralShape

::: smds.shapes.continuous_shapes.spiral_shape.SpiralShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### KleinBottleShape

::: smds.shapes.continuous_shapes.klein_bottle.KleinBottleShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### TorusShape

::: smds.shapes.continuous_shapes.torus.TorusShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

---

## Discrete Shapes

### ClusterShape

::: smds.shapes.discrete_shapes.cluster.ClusterShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### ChainShape

::: smds.shapes.discrete_shapes.chain.ChainShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### DiscreteCircularShape

::: smds.shapes.discrete_shapes.discrete_circular.DiscreteCircularShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### HierarchicalShape

::: smds.shapes.discrete_shapes.hierarchical.HierarchicalShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### GraphGeodesicShape

::: smds.shapes.discrete_shapes.graph_geodesic.GraphGeodesicShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### PolytopeShape

::: smds.shapes.discrete_shapes.polytope.PolytopeShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

---

## Spatial Shapes

### SphericalShape

::: smds.shapes.spatial_shapes.spherical.SphericalShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### CylindricalShape

::: smds.shapes.spatial_shapes.cylindrical.CylindricalShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true

### GeodesicShape

::: smds.shapes.spatial_shapes.geodesic.GeodesicShape
    options:
      members_order: source
      show_bases: true
      show_signature_annotations: true
      separate_signature: true
