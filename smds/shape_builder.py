from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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
    def _normalize_circular(values: NDArray) -> NDArray[np.float64]:
        """Normalize values to [0, 1) range for circular mapping (avoids overlap at 0 and 2π)."""
        v = np.asarray(values, dtype=np.float64).ravel()
        vmin, vmax = v.min(), v.max()
        if vmax > vmin:
            t = (v - vmin) / (vmax - vmin)
            return np.clip(t, 0, 1 - 1e-10)
        return np.zeros_like(v)

    @staticmethod
    def _category_angles(categories: NDArray) -> Tuple[NDArray, Dict]:
        """Map categories to angles, equally spaced on a circle."""
        cats = np.asarray(categories).ravel()
        unique = np.unique(cats)
        n = len(unique)
        # Map each category to an angle
        cat_to_angle = {cat: 2 * np.pi * i / n for i, cat in enumerate(unique)}
        angles = np.array([cat_to_angle[c] for c in cats])
        return angles, cat_to_angle

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
    def distinct_points(
        a_categories: NDArray,
        b_categories: NDArray,
        *,
        radius: float = 1.0,
        a_name: str = "a",
        b_name: str = "b",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Categorical + Categorical -> N*M distinct points on a circle.
        
        All points are equally distant from each other (arranged on a circle).
        
        Parameters
        ----------
        a_categories : array-like
            First categorical values.
        b_categories : array-like
            Second categorical values.
        radius : float, default=1.0
            Radius of the circle on which points are placed.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point (includes 'pair' column with combined label).
        """
        import pandas as pd
        a_grid, b_grid, df = ShapeBuilder._make_grid(a_categories, b_categories, a_name, b_name)
        
        # Create combined pair labels
        df["pair"] = [f"{a}|{b}" for a, b in zip(a_grid, b_grid)]
        
        # All unique pairs arranged equally on a circle
        unique_pairs = df["pair"].unique()
        n_pairs = len(unique_pairs)
        pair_to_angle = {pair: 2 * np.pi * i / n_pairs for i, pair in enumerate(unique_pairs)}
        
        angles = np.array([pair_to_angle[p] for p in df["pair"]])
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        Y = np.column_stack([x, y]).astype(np.float64)
        return Y, df

    @staticmethod
    def parallel_lines(
        categories: NDArray,
        line_values: NDArray,
        *,
        length: float = 2.0,
        center_radius: Optional[float] = None,
        cat_name: str = "category",
        val_name: str = "value",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Categorical + Linear -> N parallel line segments.
        
        Each category gets its own line segment. The *centers* of the line
        segments are placed equally spaced around a circle in 2D, so categories
        are maximally separated while the line directions remain parallel.
        
        Parameters
        ----------
        categories : array-like
            Categorical values (each category = one line).
        line_values : array-like
            Linear values along each line (mapped to x).
        length : float, default=2.0
            Length of each line.
        center_radius : float, optional
            Radius of the circle on which category centers are placed.
            If None, defaults to 3 * length.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        cat_grid, val_grid, df = ShapeBuilder._make_grid(categories, line_values, cat_name, val_name)

        if center_radius is None:
            center_radius = 3.0 * float(length)
        
        # Local line coordinate (all lines parallel to x-axis)
        t = ShapeBuilder._normalize(val_grid)
        x_local = float(length) * (t - 0.5)
        
        # Category centers arranged on a circle (between-category separation)
        unique_cats = np.unique(np.asarray(categories).ravel())
        n_cats = len(unique_cats)
        cat_to_phi = {cat: 2 * np.pi * i / n_cats for i, cat in enumerate(unique_cats)}
        phi = np.array([cat_to_phi[c] for c in cat_grid], dtype=np.float64)
        cx = float(center_radius) * np.cos(phi)
        cy = float(center_radius) * np.sin(phi)

        x = cx + x_local
        y = cy
        
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
        t_a = ShapeBuilder._normalize_circular(a_grid)
        theta = 2 * np.pi * t_a
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
        
        # Both axes are circular
        t_u = ShapeBuilder._normalize_circular(u_grid)
        t_v = ShapeBuilder._normalize_circular(v_grid)
        
        theta = 2 * np.pi * t_u  # Around the torus
        phi = 2 * np.pi * t_v    # Around the tube
        
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
        
        # Latitude: map to [-π/2, π/2]
        t_lat = ShapeBuilder._normalize(lat_grid)
        phi = np.pi * t_lat - np.pi / 2
        
        # Longitude: map to [0, 2π)
        t_lon = ShapeBuilder._normalize_circular(lon_grid)
        theta = 2 * np.pi * t_lon
        
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
        z = pitch * turns * t_norm
        
        Y = np.column_stack([x, y, z]).astype(np.float64)
        return Y, df

    @staticmethod
    def parallel_circles(
        categories: NDArray,
        angle_values: NDArray,
        *,
        radius: float = 1.0,
        center_radius: Optional[float] = None,
        cat_name: str = "category",
        angle_name: str = "angle",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Categorical + Circular -> N distinct circles in a circle layout (3D).
        
        Each category gets its own circle in the XY plane. Circle *centers* are
        placed equally spaced around a larger circle (also in the XY plane),
        so categories are clearly separated.
        
        Parameters
        ----------
        categories : array-like
            Categorical values (each category = one circle).
        angle_values : array-like
            Circular values around each circle.
        radius : float, default=1.0
            Radius of each circle.
        center_radius : float, optional
            Radius of the circle on which category centers are placed.
            If None, defaults to 3 * radius.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 3)
            3D coordinates for each point.
        labels : DataFrame
            Labels for each point.
        """
        cat_grid, angle_grid, df = ShapeBuilder._make_grid(categories, angle_values, cat_name, angle_name)
        
        if center_radius is None:
            center_radius = 3.0 * float(radius)

        # Angles along each circle (within-category)
        t_a = ShapeBuilder._normalize_circular(angle_grid)
        theta = 2 * np.pi * t_a

        # Category centers arranged on a circle (between-category separation)
        unique_cats = np.unique(np.asarray(categories).ravel())
        n_cats = len(unique_cats)
        cat_to_phi = {cat: 2 * np.pi * i / n_cats for i, cat in enumerate(unique_cats)}
        phi = np.array([cat_to_phi[c] for c in cat_grid], dtype=np.float64)
        cx = float(center_radius) * np.cos(phi)
        cy = float(center_radius) * np.sin(phi)

        # Local circle around each center (in the XY plane)
        x = cx + float(radius) * np.cos(theta)
        y = cy + float(radius) * np.sin(theta)
        z = np.zeros_like(x)

        Y = np.column_stack([x, y, z]).astype(np.float64)
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
        
        t_a = ShapeBuilder._normalize_circular(angle_grid)
        theta = 2 * np.pi * t_a
        
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

