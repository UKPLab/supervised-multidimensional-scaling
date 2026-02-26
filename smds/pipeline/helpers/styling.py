"""
Centralized styling definitions for SMDS visualizations.

This module defines the color schemes used to distinguish between different
categories of Manifold Shapes (Continuous, Discrete, Spatial, Special).
It ensures consistency across different plots and visualizations.
"""

# Continuous: Blue
COL_CONTINUOUS = "#3498db"
# Discrete: Orange
COL_DISCRETE = "#e67e22"
# Spatial: Green
COL_SPATIAL = "#2ecc71"

# Fallback color for unknown shapes
COL_DEFAULT = "#95a5a6"

COL_USER_PROVIDED = "#9b59b6"

SHAPE_COLORS = {
    # Continuous Shapes
    "CircularShape": COL_CONTINUOUS,
    "EuclideanShape": COL_CONTINUOUS,
    "LogLinearShape": COL_CONTINUOUS,
    "SemicircularShape": COL_CONTINUOUS,
    "SpiralShape": COL_CONTINUOUS,
    # Discrete Shapes
    "ChainShape": COL_DISCRETE,
    "ClusterShape": COL_DISCRETE,
    "DiscreteCircularShape": COL_DISCRETE,
    "HierarchicalShape": COL_DISCRETE,
    # Spatial Shapes
    "CylindricalShape": COL_SPATIAL,
    "GeodesicShape": COL_SPATIAL,
    "SphericalShape": COL_SPATIAL,
}


def get_shape_color(shape_name: str) -> str:
    """
    Retrieve the hex color code for a specific shape class.

    Args:
        shape_name (str): The class name of the shape (e.g., "CircularShape").

    Returns
    -------
        str: The corresponding hex color, or a default grey if not found.
    """
    known = SHAPE_COLORS.get(shape_name)
    return known if known is not None else COL_USER_PROVIDED
