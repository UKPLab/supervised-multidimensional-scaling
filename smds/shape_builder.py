from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import plotly.graph_objects as go

class ShapeBuilder:
    """
    Simple, explicit shape builder for creating target embedding matrices.
    
    Each method creates the full Cartesian product of label values and returns
    (Y, labels_df) where:
    - Y: NDArray of shape (n_points, n_dims) with coordinates
    - labels_df: DataFrame with label columns for each point
    
    All categorical combinations place clusters equally distant from each other
    (on a circle) for maximum separation.
    
    Examples
    --------
    >>> # Create a cylinder from year (linear) and month (circular)
    >>> years = np.arange(2010, 2020)
    >>> months = np.arange(1, 13)
    >>> Y, labels = ShapeBuilder.cylinder(years, months)
    >>> Y.shape  # (120, 3) - 10 years × 12 months, 3D coordinates
    
    >>> # Create parallel circles for categories
    >>> colors = ['red', 'blue', 'green']
    >>> angles = np.linspace(0, 1, 36)
    >>> Y, labels = ShapeBuilder.parallel_circles(colors, angles)
    """

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_grid(a_values: NDArray, b_values: NDArray, a_name: str = "a", b_name: str = "b"):
        """Create Cartesian product grid of two value arrays."""
        import pandas as pd
        a = np.asarray(a_values).ravel()
        b = np.asarray(b_values).ravel()
        a_grid = np.repeat(a, len(b))
        b_grid = np.tile(b, len(a))
        df = pd.DataFrame({a_name: a_grid, b_name: b_grid})
        return a_grid, b_grid, df

    @staticmethod
    def _normalize(values: NDArray) -> NDArray[np.float64]:
        """Normalize values to [0, 1] range."""
        v = np.asarray(values, dtype=np.float64).ravel()
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            return (v - vmin) / (vmax - vmin)
        return np.zeros_like(v)

    @staticmethod
    def _map_to_circle(values: NDArray) -> Tuple[NDArray, Dict]:
        """Map categories to angles, equally spaced on a circle."""
        vals = np.asarray(values).ravel()
        unique = np.unique(vals)
        n = len(unique)
        # Map each category to an angle
        val_to_angle = {val: 2 * np.pi * i / n for i, val in enumerate(unique)}
        angles = np.array([val_to_angle[v] for v in vals], dtype=np.float64)
        return angles, val_to_angle
    
    @staticmethod
    def _map_to_half_circle(values: NDArray) -> NDArray:
        """Map discrete values to evenly-spaced angles on a semicircle."""
        vals = np.asarray(values).ravel()
        t = ShapeBuilder._normalize(vals)
        angles = np.pi * t
        return angles
    
    @staticmethod
    def _get_cluster_centers(n_clusters: int, outer_radius: float = 3.0) -> NDArray[np.float64]:
        """
        Generate cluster center positions using Fibonacci sphere.
        """
        indices = np.arange(n_clusters, dtype=np.float64)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_clusters)
        theta = np.pi * (1 + 5**0.5) * indices

        x = outer_radius * np.sin(phi) * np.cos(theta)
        y = outer_radius * np.sin(phi) * np.sin(theta)
        z = outer_radius * np.cos(phi)

        return np.column_stack([x, y, z])

    # -------------------------------------------------------------------------
    # 2D Shapes
    # -------------------------------------------------------------------------

    @staticmethod
    def plane(
        a_values: NDArray,
        b_values: NDArray,
        *,
        width: float = 2.0,
        height: float = 2.0,
        a_name: str = "a",
        b_name: str = "b",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Linear + Linear -> 2D plane (grid of points).
        
        Parameters
        ----------
        a_values : array-like
            Values for the first linear axis (mapped to x).
        b_values : array-like
            Values for the second linear axis (mapped to y).
        width : float, default=2.0
            Total width of the plane (x range: -width/2 to +width/2).
        height : float, default=2.0
            Total height of the plane (y range: -height/2 to +height/2).
        
        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        a_grid, b_grid, df = ShapeBuilder._make_grid(a_values, b_values, a_name, b_name)
        
        # Normalize to [0, 1] then scale
        t_a = ShapeBuilder._normalize(a_grid)
        t_b = ShapeBuilder._normalize(b_grid)
        
        x = width * (t_a - 0.5)
        y = height * (t_b - 0.5)
        
        Y = np.column_stack([x, y]).astype(np.float64)
        return Y, df
    
    
    @staticmethod
    def clusters_of_circles(
        categories: NDArray,
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        outer_radius: float = 3.0,
        cat_name: str = "category",
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], pd.DataFrame]:
        """
        Categorical + Circular -> Circles at cluster positions.

        - 2-3 clusters: positioned in 2D
        - 4+ clusters: positioned in 3D (Fibonacci sphere)
        - Circles in local XY plane
        """
        cat_grid, angle_grid, df = ShapeBuilder._make_grid(categories, angle_values, cat_name, angle_name)

        unique_cats = np.unique(np.asarray(categories).ravel())
        n_clusters = len(unique_cats)
        cluster_centers_3d = ShapeBuilder._get_cluster_centers(n_clusters, outer_radius)

        cat_to_idx = {val: i for i, val in enumerate(unique_cats)}
        cluster_indices = np.array([cat_to_idx[val] for val in cat_grid])
        centers = cluster_centers_3d[cluster_indices]

        theta, _ = ShapeBuilder._map_to_circle(angle_grid)
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        dz = np.zeros_like(dx)

        positions_3d = centers + np.column_stack([dx, dy, dz])
        Y = positions_3d.astype(np.float64)

        return Y, df

    @staticmethod
    def circle_of_circles(
        outer_angle_values: NDArray,
        inner_angle_values: NDArray,
        *,
        outer_radius: float = 2.5,
        inner_radius: float = 0.6,
        outer_name: str = "outer_angle",
        inner_name: str = "inner_angle",
    ) -> Tuple[NDArray[np.float64], pd.DataFrame]:
        """
        Circular + Circular -> 2D circle of circles.

        Places the center of a small circle at each point on a larger outer circle.
        The final coordinates are in 2D:
            center(theta) + inner_circle(phi)

        Parameters
        ----------
        outer_angle_values : array-like
            Values mapped to the outer circle angle (theta).
        inner_angle_values : array-like
            Values mapped to each local inner circle angle (phi).
        outer_radius : float, default=2.5
            Radius of the outer center circle.
        inner_radius : float, default=0.6
            Radius of each local circle.
        outer_name : str, default="outer_angle"
            Column name for outer-angle labels.
        inner_name : str, default="inner_angle"
            Column name for inner-angle labels.

        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        outer_grid, inner_grid, df = ShapeBuilder._make_grid(
            outer_angle_values, inner_angle_values, outer_name, inner_name
        )

        theta, _ = ShapeBuilder._map_to_circle(outer_grid)
        phi, _ = ShapeBuilder._map_to_circle(inner_grid)

        cx = float(outer_radius) * np.cos(theta)
        cy = float(outer_radius) * np.sin(theta)

        dx = float(inner_radius) * np.cos(phi)
        dy = float(inner_radius) * np.sin(phi)

        x = cx + dx
        y = cy + dy

        Y = np.column_stack([x, y]).astype(np.float64)
        return Y, df


    @staticmethod
    def clusters_of_lines(
        categories: NDArray,
        line_values: NDArray,
        *,
        length: float = 2.0,
        outer_radius: float = 3.0,
        cat_name: str = "category",
        val_name: str = "value",
    ) -> Tuple[NDArray[np.float64], pd.DataFrame]:
        """
        Categorical + Linear -> Lines at cluster positions.

        - 2-3 clusters: positioned in 2D
        - 4+ clusters: positioned in 3D (Fibonacci sphere)
        - Lines parallel to local x-axis
        """
        cat_grid, val_grid, df = ShapeBuilder._make_grid(categories, line_values, cat_name, val_name)

        unique_cats = np.unique(np.asarray(categories).ravel())
        n_clusters = len(unique_cats)
        cluster_centers_3d = ShapeBuilder._get_cluster_centers(n_clusters, outer_radius)

        cat_to_idx = {val: i for i, val in enumerate(unique_cats)}
        cluster_indices = np.array([cat_to_idx[val] for val in cat_grid])
        centers = cluster_centers_3d[cluster_indices]

        t = ShapeBuilder._normalize(val_grid)
        x_local = length * (t - 0.5)

        positions_3d = centers + np.column_stack([x_local, np.zeros_like(x_local), np.zeros_like(x_local)])

        Y = positions_3d.astype(np.float64)

        return Y, df
    

    @staticmethod
    def half_circle(
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Circular (semicircle) -> 2D semicircle arc.
        
        Maps values to evenly-spaced points on a semicircular arc from
        0° to 180° (left to right).
        
        Parameters
        ----------
        angle_values : array-like
            Discrete values to map around the semicircle.
        radius : float, default=1.0
            Radius of the semicircle.
        angle_name : str, default="angle"
            Name for the angle column in the labels DataFrame.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        
        Examples
        --------
        >>> # Map 7 days of week to a semicircle
        >>> days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        >>> Y, labels = ShapeBuilder.half_circle(days)
        >>> Y.shape  # (7, 2)
        """
        import pandas as pd
        angles = np.asarray(angle_values).ravel()
        df = pd.DataFrame({angle_name: angles})
        
        # Map to semicircle (0 to π)
        theta = ShapeBuilder._map_to_half_circle(angles)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        Y = np.column_stack([x, y]).astype(np.float64)
        return Y, df


    # -------------------------------------------------------------------------
    # 3D Shapes
    # -------------------------------------------------------------------------

    @staticmethod
    def cylinder(
        height_values: NDArray,
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        height: float = 2.0,
        height_name: str = "height",
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Linear + Circular -> 3D cylinder surface.
        
        Parameters
        ----------
        height_values : array-like
            Linear values mapped to cylinder height (z-axis).
        angle_values : array-like
            Circular values mapped to angle around cylinder.
        radius : float, default=1.0
            Cylinder radius.
        height : float, default=2.0
            Cylinder height (z range: -height/2 to +height/2).
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        h_grid, a_grid, df = ShapeBuilder._make_grid(height_values, angle_values, height_name, angle_name)
        
        # Height: linear mapping
        t_h = ShapeBuilder._normalize(h_grid)
        z = height * (t_h - 0.5)
        
        # Angle: circular mapping
        theta, _ = ShapeBuilder._map_to_circle(a_grid)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df
    
    @staticmethod
    def half_cylinder(
        height_values: NDArray,
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        height: float = 2.0,
        height_name: str = "height",
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Linear + Circular (semicircle) -> 3D half-cylinder surface.
        
        Creates a half-cylinder (semicircular cross-section) with height
        along the z-axis and semicircular arc in the xy-plane.
        
        Parameters
        ----------
        height_values : array-like
            Linear values mapped to cylinder height (z-axis).
        angle_values : array-like
            Discrete values mapped to semicircular arc (0° to 180°).
        radius : float, default=1.0
            Radius of the half-cylinder.
        height : float, default=2.0
            Height of the half-cylinder (z range: -height/2 to +height/2).
        height_name : str, default="height"
            Name for the height column in labels.
        angle_name : str, default="angle"
            Name for the angle column in labels.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        
        Examples
        --------
        >>> # Half-cylinder for years × months
        >>> years = np.arange(2020, 2025)
        >>> months = np.arange(1, 13)
        >>> Y, labels = ShapeBuilder.half_cylinder(years, months)
        >>> Y.shape  # (60, 3) - 5 years × 12 months
        """
        h_grid, a_grid, df = ShapeBuilder._make_grid(height_values, angle_values, height_name, angle_name)
        
        # Height: linear mapping
        t_h = ShapeBuilder._normalize(h_grid)
        z = height * (t_h - 0.5)
        
        # Angle: semicircular mapping (0 to π)
        theta = ShapeBuilder._map_to_half_circle(a_grid)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df

    @staticmethod
    def torus(
        u_values: NDArray,
        v_values: NDArray,
        *,
        major_radius: float = 2.0,
        minor_radius: float = 1.0,
        u_name: str = "u",
        v_name: str = "v",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Circular + Circular -> 3D torus surface.
        
        Parameters
        ----------
        u_values : array-like
            First circular values (angle around the torus ring).
        v_values : array-like
            Second circular values (angle around the tube).
        major_radius : float, default=2.0
            Distance from torus center to tube center.
        minor_radius : float, default=1.0
            Radius of the tube.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        u_grid, v_grid, df = ShapeBuilder._make_grid(u_values, v_values, u_name, v_name)
        
        # Both axes: map to circle (no overlap)
        theta, _ = ShapeBuilder._map_to_circle(u_grid)  # Around the torus
        phi, _ = ShapeBuilder._map_to_circle(v_grid)    # Around the tube
        
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df


    @staticmethod
    def sphere(
        lat_values: NDArray,
        lon_values: NDArray,
        *,
        radius: float = 1.0,
        lat_name: str = "latitude",
        lon_name: str = "longitude",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Two circular features -> 3D sphere surface.
        
        Uses latitude/longitude parameterization:
        - Latitude: -90° (south pole) to +90° (north pole)
        - Longitude: 0° to 360° around the equator
        
        Parameters
        ----------
        lat_values : array-like
            Latitude values (mapped to -π/2 to +π/2).
        lon_values : array-like
            Longitude values (mapped to 0 to 2π).
        radius : float, default=1.0
            Sphere radius.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        lat_grid, lon_grid, df = ShapeBuilder._make_grid(lat_values, lon_values, lat_name, lon_name)
    
        # Latitude mapping
        t_lat = ShapeBuilder._normalize(lat_grid)

        epsilon = 0.05  # Small offset from poles to avoid poles
        phi = np.pi * (t_lat * (1 - 2*epsilon) + epsilon) - np.pi / 2
        
        # Longitude mapping
        theta, _ = ShapeBuilder._map_to_circle(lon_grid)
        
        x = radius * np.cos(phi) * np.cos(theta)
        y = radius * np.cos(phi) * np.sin(theta)
        z = radius * np.sin(phi)
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df

    @staticmethod
    def helix(
        t_values: NDArray,
        *,
        radius: float = 1.0,
        pitch: float = 1.0,
        turns: float = 2.0,
        t_name: str = "t",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Single linear feature -> 3D helix (spiral) curve.
        
        Parameters
        ----------
        t_values : array-like
            Linear values along the helix.
        radius : float, default=1.0
            Helix radius.
        pitch : float, default=1.0
            Vertical distance per turn.
        turns : float, default=2.0
            Number of complete rotations.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        import pandas as pd
        t = np.asarray(t_values).ravel()
        df = pd.DataFrame({t_name: t})
        
        # Normalize t to [0, 1]
        t_norm = ShapeBuilder._normalize(t)

        theta = 2 * np.pi * turns * t_norm
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = pitch * turns * (t_norm - 0.5)  # Centered around zero
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df
    
    @staticmethod
    def helix_plus_line(
        line_values: NDArray,
        helix_t_values: NDArray,
        *,
        line_spacing: float = 1.0,
        helix_radius: float = 0.3,
        helix_pitch: float = 1.0,
        helix_turns: float = 2.0,
        line_name: str = "line",
        helix_name: str = "helix_t",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Linear + Helix -> 3D shape: parallel corkscrews extending from a
        base line.

        The base line runs along x. At each line position a helix extends
        outward (perpendicular to the line) along y, spiraling in the x-z
        plane.  A true 3D helix requires 3 independent axes — 2 for the
        circular spiral and 1 for advancement — so the spiral necessarily
        shares x with the line, adding a small radial oscillation
        (``helix_radius``) around each line position.

        Parameters
        ----------
        line_values : array-like
            Values along the base line (x-axis positions).
        helix_t_values : array-like
            Parameter values along each helix (0 to 1 or similar).
        line_spacing : float, default=1.0
            Distance between consecutive points on the base line.
        helix_radius : float, default=0.3
            Radius of each helix spiral.  Kept smaller than
            ``line_spacing / 2`` to avoid overlap between adjacent helices.
        helix_pitch : float, default=1.0
            Distance per complete helix turn along y.
        helix_turns : float, default=2.0
            Number of complete rotations for each helix.
        line_name : str, default="line"
            Name for the line position column in labels.
        helix_name : str, default="helix_t"
            Name for the helix parameter column in labels.

        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.

        Examples
        --------
        >>> # Create 5 helices along a line
        >>> line_positions = np.arange(5)
        >>> helix_param = np.linspace(0, 10, 50)
        >>> Y, labels = ShapeBuilder.helix_plus_line(line_positions, helix_param)
        >>> plot_shape(Y, labels, color_by='line')
        """
        line_grid, helix_grid, df = ShapeBuilder._make_grid(
            line_values, helix_t_values, line_name, helix_name
        )

        # Base line: position along x-axis (centered)
        t_line = ShapeBuilder._normalize(line_grid)
        n_line = len(np.unique(line_values))
        total_length = line_spacing * (n_line - 1) if n_line > 1 else 0
        x_base = line_spacing * t_line * (n_line - 1) - total_length / 2

        # Helix parameter and angle
        t_helix = ShapeBuilder._normalize(helix_grid)

        theta = 2 * np.pi * helix_turns * t_helix

        # Clamp radius so adjacent helices never overlap
        safe_radius = min(helix_radius, line_spacing / 2 * 0.9) if n_line > 1 else helix_radius

        # Spiral in x-z plane (shares x with line — small oscillation)
        x = x_base + safe_radius * np.cos(theta)
        z = safe_radius * np.sin(theta)

        # Advancement along y (perpendicular to line), centered at zero
        y = helix_pitch * helix_turns * (t_helix - 0.5)

        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df


    @staticmethod
    def clusters_of_clusters(
        a_categories: NDArray,
        b_categories: NDArray,
        *,
        outer_radius: float = 3.0,
        inner_radius: float = 1.0,
        a_name: str = "a",
        b_name: str = "b",
    ) -> Tuple[NDArray[np.float64], pd.DataFrame]:
        """
        Categorical + Categorical -> Clusters of clusters.

        - 2-3 clusters: positioned in 2D (line or triangle)
        - 4+ clusters: positioned in 3D (Fibonacci sphere)
        - Inner points within each cluster placed on a small circle
        """
        a_grid, b_grid, df = ShapeBuilder._make_grid(a_categories, b_categories, a_name, b_name)
        df["pair"] = [f"{a}|{b}" for a, b in zip(a_grid, b_grid)]

        unique_a = np.unique(np.asarray(a_categories).ravel())
        n_clusters = len(unique_a)
        cluster_centers_3d = ShapeBuilder._get_cluster_centers(n_clusters, outer_radius)

        a_to_idx = {val: i for i, val in enumerate(unique_a)}
        cluster_indices = np.array([a_to_idx[val] for val in a_grid])
        centers = cluster_centers_3d[cluster_indices]

        inner_angles, _ = ShapeBuilder._map_to_circle(b_grid)
        dx = inner_radius * np.cos(inner_angles)
        dy = inner_radius * np.sin(inner_angles)
        dz = np.zeros_like(dx)

        positions_3d = centers + np.column_stack([dx, dy, dz])

        Y = positions_3d.astype(np.float64)

        return Y, df

    @staticmethod
    def parallel_cylinders(
        categories: NDArray,
        height_values: NDArray,
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        height: float = 2.0,
        center_radius: Optional[float] = None,
        cat_name: str = "category",
        height_name: str = "height",
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Categorical + Cylinder -> N parallel cylinders in 3D.
        
        Each category gets its own cylinder (axis aligned with z). Cylinder
        *centers* are placed equally spaced around a circle in the XY plane,
        so categories are maximally separated.
        
        Parameters
        ----------
        categories : array-like
            Categorical values (each category = one cylinder).
        height_values : array-like
            Linear values along each cylinder (z-axis).
        angle_values : array-like
            Circular values around each cylinder.
        radius : float, default=1.0
            Radius of each cylinder.
        height : float, default=2.0
            Height of each cylinder.
        center_radius : float, optional
            Radius of the circle on which category cylinder centers are placed.
            If None, defaults to 4 * radius.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        import pandas as pd
        
        cats = np.asarray(categories).ravel()
        heights = np.asarray(height_values).ravel()
        angles = np.asarray(angle_values).ravel()

        if center_radius is None:
            center_radius = 4.0 * float(radius)
        
        # Triple Cartesian product: category × height × angle
        n_cats = len(cats)
        n_heights = len(heights)
        n_angles = len(angles)
        
        cat_grid = np.repeat(cats, n_heights * n_angles)
        height_grid = np.tile(np.repeat(heights, n_angles), n_cats)
        angle_grid = np.tile(angles, n_cats * n_heights)
        
        df = pd.DataFrame({
            cat_name: cat_grid,
            height_name: height_grid,
            angle_name: angle_grid,
        })
        
        # Cylinder coordinates
        t_h = ShapeBuilder._normalize(height_grid)
        z = height * (t_h - 0.5)
        
        theta, _ = ShapeBuilder._map_to_circle(angle_grid)
        
        # Base cylinder at origin
        x_cyl = float(radius) * np.cos(theta)
        y_cyl = float(radius) * np.sin(theta)
        
        # Category centers arranged on a circle (between-category separation)
        unique_cats = np.unique(cats)
        n_unique = len(unique_cats)
        cat_to_phi = {cat: 2 * np.pi * i / n_unique for i, cat in enumerate(unique_cats)}
        phi = np.array([cat_to_phi[c] for c in cat_grid], dtype=np.float64)
        cx = float(center_radius) * np.cos(phi)
        cy = float(center_radius) * np.sin(phi)

        x = x_cyl + cx
        y = y_cyl + cy
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df

    # -------------------------------------------------------------------------
    # Lookup method for mapping samples to grid
    # -------------------------------------------------------------------------

    @staticmethod
    def lookup(
        labels_df: "pd.DataFrame",
        Y: NDArray[np.float64],
        **sample_labels: NDArray,
    ) -> NDArray[np.float64]:
        """
        Map per-sample labels to coordinates in a precomputed grid.
        
        Parameters
        ----------
        labels_df : DataFrame
            Labels grid from a ShapeBuilder method.
        Y : NDArray
            Coordinates grid from a ShapeBuilder method.
        **sample_labels : NDArray
            Label arrays for each sample (one per column in labels_df, 
            excluding 'pair' column if present).
        
        Returns
        -------
        Y_samples : NDArray
            Coordinates for each sample.
        
        Example
        -------
        >>> Y_grid, labels_grid = ShapeBuilder.cylinder(years, months)
        >>> Y_samples = ShapeBuilder.lookup(labels_grid, Y_grid, 
        ...                                  height=sample_years, angle=sample_months)
        """
        import pandas as pd
        
        # Get columns to match (exclude 'pair' if present)
        cols = [c for c in labels_df.columns if c != "pair"]
        
        missing = [c for c in cols if c not in sample_labels]
        if missing:
            raise ValueError(f"Missing label arrays: {missing}")
        
        # Validate sample lengths
        n_samples = len(sample_labels[cols[0]])
        for c in cols[1:]:
            if len(sample_labels[c]) != n_samples:
                raise ValueError("All sample label arrays must have the same length.")
        
        # Build lookup index
        grid_cols = [labels_df[c].values for c in cols]
        index: Dict[Tuple, int] = {}
        for i in range(len(labels_df)):
            key = tuple(col[i] for col in grid_cols)
            index[key] = i
        
        # Lookup each sample
        sample_cols = [np.asarray(sample_labels[c]) for c in cols]
        Y_samples = np.empty((n_samples, Y.shape[1]), dtype=np.float64)
        
        for i in range(n_samples):
            key = tuple(col[i] for col in sample_cols)
            if key not in index:
                raise KeyError(f"Label combination {key} not found in grid.")
            Y_samples[i] = Y[index[key]]
        
        return Y_samples




def plot_shape(
    Y: NDArray[np.float64],
    labels_df: pd.DataFrame,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    size: int = 8,
    opacity: float = 0.7,
    output_file: Optional[str] = None,
) -> go.Figure:
    """Plot shape with interactive 3D/2D visualization."""

    n_dims = Y.shape[1]

    if color_by is None:
        color_by = labels_df.columns[0]

    x, y = Y[:, 0], Y[:, 1]
    z = Y[:, 2] if n_dims == 3 else None

    color_data = labels_df[color_by].values
    is_categorical = labels_df[color_by].dtype == object or pd.api.types.is_categorical_dtype(labels_df[color_by])

    if is_categorical:
        unique_vals = pd.unique(color_data)
        color_map = {val: i for i, val in enumerate(unique_vals)}
        color_numeric = np.array([color_map[val] for val in color_data])
        colorscale = "Viridis"
        colorbar_title = color_by
        tickvals = list(range(len(unique_vals)))
        ticktext = [str(v) for v in unique_vals]
    else:
        color_numeric = color_data.astype(float)
        colorscale = "Viridis"
        colorbar_title = color_by
        tickvals = None
        ticktext = None

    hover_parts = []
    for col in labels_df.columns:
        hover_parts.append(f"{col}=%{{customdata[{list(labels_df.columns).index(col)}]}}")
    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

    customdata = labels_df.values

    marker_common = dict(
        size=size,
        opacity=opacity,
        color=color_numeric,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=colorbar_title,
            tickvals=tickvals,
            ticktext=ticktext,
        ) if is_categorical and tickvals else dict(title=colorbar_title),
    )

    if n_dims == 3:
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=marker_common,
            showlegend=False,
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
        )
    else:
        trace = go.Scatter(
            x=x, y=y,
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=marker_common,
            showlegend=False,
        )
        fig = go.Figure(data=[trace])
        fig.update_layout(
            xaxis=dict(title="X", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y"),
        )
    

    if title is None:
        label_cols = [col for col in labels_df.columns if col != 'pair']
        title = f"Shape: {' × '.join(label_cols)}"

    fig.update_layout(
        title=title,
        width=800,
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    if output_file:
        fig.write_html(output_file, include_plotlyjs="cdn")
        print(f"  ✓ {output_file}")

    return fig

def main():
    """Generate all HTML visualizations."""

    print("="*80)
    print("ShapeBuilder - Generating HTML Visualizations")
    print("="*80)

    # Test configurations
    test_configs = [
        (2, "2_clusters"),
        (3, "3_clusters"),
        (4, "4_clusters"),
        (5, "5_clusters"),
        (6, "6_clusters"),
        (8, "8_clusters"),
    ]

    inner_categories = ['X', 'Y', 'Z']
    line_points = np.linspace(0, 10, 7)
    circle_angles = np.arange(8)

    # 1. Clusters of Clusters
    print("\n[1/4] Clusters of Clusters (Categorical + Categorical)")
    print("-"*80)
    for n_clusters, label in test_configs:
        cats = [chr(65+i) for i in range(n_clusters)]  # A, B, C, ...
        Y, labels = ShapeBuilder.clusters_of_clusters(
            cats, inner_categories, outer_radius=3.0, inner_radius=0.5
        )

        filename = f"clusters_of_clusters_{label}.html"
        plot_shape(
            Y, labels, color_by='a',
            title=f"Clusters of Clusters: {n_clusters} × 3 (3D)",
            output_file=filename, size=10
        )

    # 2. Clusters of Lines
    print("\n[2/4] Clusters of Lines (Categorical + Linear)")
    print("-"*80)
    for n_clusters, label in test_configs:
        cats = [chr(65+i) for i in range(n_clusters)]
        Y, labels = ShapeBuilder.clusters_of_lines(
            cats, line_points, length=2.0, outer_radius=3.0
        )

        filename = f"clusters_of_lines_{label}.html"
        plot_shape(
            Y, labels, color_by='category',
            title=f"Clusters of Lines: {n_clusters} × 7 (3D)",
            output_file=filename, size=10
        )

    # 3. Clusters of Circles
    print("\n[3/4] Clusters of Circles (Categorical + Circular)")
    print("-"*80)
    for n_clusters, label in test_configs:
        cats = [chr(65+i) for i in range(n_clusters)]
        Y, labels = ShapeBuilder.clusters_of_circles(
            cats, circle_angles, radius=0.8, outer_radius=3.0
        )

        filename = f"clusters_of_circles_{label}.html"
        plot_shape(
            Y, labels, color_by='category',
            title=f"Clusters of Circles: {n_clusters} × 8 (3D)",
            output_file=filename, size=10
        )

    # 4. Circle of Circles (2D)
    print("\n[4/5] Circle of Circles (Circular + Circular, 2D)")
    print("-"*80)
    coc_configs = [
        (8, 6, "8x6", 2.3, 0.5),
        (10, 8, "10x8", 2.8, 0.6),
        (12, 10, "12x10", 3.2, 0.7),
    ]
    for n_outer, n_inner, label, outer_r, inner_r in coc_configs:
        outer_angles = np.arange(n_outer)
        inner_angles = np.arange(n_inner)

        Y, labels = ShapeBuilder.circle_of_circles(
            outer_angles,
            inner_angles,
            outer_radius=outer_r,
            inner_radius=inner_r,
        )

        filename = f"circle_of_circles_{label}.html"
        plot_shape(
            Y,
            labels,
            color_by='outer_angle',
            title=f"Circle of Circles: outer={n_outer}, inner={n_inner} (2D)",
            output_file=filename,
            size=8,
        )

    # 5. Helix + Line
    print("\n[5/5] Helix + Line (Linear + Helix)")
    print("-"*80)
    
    helix_configs = [
        (3, 30, "3_helices", 2.5, 0.4, 1.0, 2.0),
        (5, 40, "5_helices", 2.0, 0.5, 1.0, 3.0),
        (7, 50, "7_helices", 1.5, 0.4, 0.8, 2.5),
        (4, 60, "4_helices_dense", 2.0, 0.6, 1.5, 4.0),
        (6, 35, "6_helices_tight", 1.8, 0.3, 0.6, 2.0),
        (8, 45, "8_helices", 1.5, 0.4, 1.0, 2.5),
    ]
    
    for n_lines, n_helix_points, label, spacing, radius, pitch, turns in helix_configs:
        line_positions = np.arange(n_lines)
        helix_param = np.linspace(0, 10, n_helix_points)
        
        Y, labels = ShapeBuilder.helix_plus_line(
            line_positions,
            helix_param,
            line_spacing=spacing,
            helix_radius=radius,
            helix_pitch=pitch,
            helix_turns=turns
        )
        
        filename = f"helix_plus_line_{label}.html"
        plot_shape(
            Y, labels,
            color_by='line',
            title=f"Helix + Line: {n_lines} helices × {n_helix_points} points (spacing={spacing}, turns={turns})",
            output_file=filename,
            size=8
        )

    print("\n" + "="*80)
    print(f"✓ Generated {len(test_configs) * 3 + len(coc_configs) + len(helix_configs)} HTML visualizations")
    print("="*80)
    print("\nFiles created:")
    print("  - clusters_of_clusters_{2,3,4,5,6,8}_clusters.html")
    print("  - clusters_of_lines_{2,3,4,5,6,8}_clusters.html")
    print("  - clusters_of_circles_{2,3,4,5,6,8}_clusters.html")
    print("  - circle_of_circles_{8x6,10x8,12x10}.html")
    print("  - helix_plus_line_{3,5,7,4_dense,6_tight,8}_helices.html")

if __name__ == "__main__":
    main()