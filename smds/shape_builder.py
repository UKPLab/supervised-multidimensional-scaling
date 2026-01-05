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
        outer_radius: float = 3.0,
        inner_radius: float = 1.0,
        a_name: str = "a",
        b_name: str = "b",
    ) -> Tuple[NDArray[np.float64], "pd.DataFrame"]:
        """
        Categorical + Categorical -> Nested clusters (maximum distance grouping).
        
        Creates a two-level hierarchy:
        - Outer category (a_categories): clusters placed on a large circle
        - Inner category (b_categories): points within each cluster on a smaller circle
        
        This ensures both categorical levels are maximally separated at their
        respective scales.
        
        Parameters
        ----------
        a_categories : array-like
            Outer categorical values (each gets a cluster).
        b_categories : array-like
            Inner categorical values (points within each cluster).
        outer_radius : float, default=3.0
            Radius of the circle on which outer category centers are placed.
        inner_radius : float, default=1.0
            Radius of the small circle for points within each cluster.
        
        Returns
        -------
        Y : NDArray of shape (n_points, 2)
            2D coordinates for each point.
        labels : DataFrame
            Labels for each point (includes 'pair' column with combined label).
        
        Examples
        --------
        >>> # A/B with 1/2/3: creates 2 clusters with 3 points each
        >>> Y, labels = ShapeBuilder.distinct_points(['A', 'B'], [1, 2, 3])
        >>> Y.shape  # (6, 2)
        """
        import pandas as pd
        a_grid, b_grid, df = ShapeBuilder._make_grid(a_categories, b_categories, a_name, b_name)
        
        # Create combined pair labels
        df["pair"] = [f"{a}|{b}" for a, b in zip(a_grid, b_grid)]
        
        # Outer category: place cluster centers on a large circle
        unique_outer = np.unique(np.asarray(a_categories).ravel())
        n_outer = len(unique_outer)
        outer_to_angle = {cat: 2 * np.pi * i / n_outer for i, cat in enumerate(unique_outer)}
        
        # Inner category: place points within each cluster on a small circle
        unique_inner = np.unique(np.asarray(b_categories).ravel())
        n_inner = len(unique_inner)
        inner_to_angle = {cat: 2 * np.pi * i / n_inner for i, cat in enumerate(unique_inner)}
        
        # Calculate positions
        x = np.zeros(len(df), dtype=np.float64)
        y = np.zeros(len(df), dtype=np.float64)
        
        for i, (a_val, b_val) in enumerate(zip(a_grid, b_grid)):
            # Outer category determines cluster center
            outer_angle = outer_to_angle[a_val]
            cx = outer_radius * np.cos(outer_angle)
            cy = outer_radius * np.sin(outer_angle)
            
            # Inner category determines position within cluster
            inner_angle = inner_to_angle[b_val]
            dx = inner_radius * np.cos(inner_angle)
            dy = inner_radius * np.sin(inner_angle)
            
            # Final position: cluster center + offset
            x[i] = cx + dx
            y[i] = cy + dy
        
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
    
        # Ensure sufficient separation between circles
        unique_cats = np.unique(np.asarray(categories).ravel())
        n_cats = len(unique_cats)
        
        if center_radius is None:
            # Increase separation: ensure circles don't overlap
            # Need center_radius > radius * (1 + 2*sin(π/n_cats)) for no overlap
            min_separation = radius * (1 + 2 * np.sin(np.pi / n_cats))
            center_radius = max(4.0 * float(radius), min_separation * 1.5)

        # Angles along each circle (within-category)
        t_a = ShapeBuilder._normalize_circular(angle_grid)
        theta = 2 * np.pi * t_a

        # Category centers arranged on a circle
        # Offset by π/n_cats to improve visual balance
        cat_to_phi = {cat: 2 * np.pi * i / n_cats + np.pi / n_cats 
                    for i, cat in enumerate(unique_cats)}
        phi = np.array([cat_to_phi[c] for c in cat_grid], dtype=np.float64)
        cx = float(center_radius) * np.cos(phi)
        cy = float(center_radius) * np.sin(phi)

        # Local circle around each center
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




def plot_shape(
    Y: NDArray[np.float64],
    labels_df: pd.DataFrame,
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    size: int = 8,
    opacity: float = 0.7,
    output_file: Optional[str] = None,
    show: bool = True,
) -> go.Figure:
    """
    Plot any ShapeBuilder output with interactive 3D/2D visualization.
    
    Parameters
    ----------
    Y : NDArray
        Coordinate array from ShapeBuilder (shape: n_points × 2 or 3).
    labels_df : DataFrame
        Labels DataFrame from ShapeBuilder.
    color_by : str, optional
        Column name in labels_df to use for coloring points.
        If None, uses the first column.
    title : str, optional
        Plot title. If None, auto-generated from labels.
    size : int, default=8
        Marker size.
    opacity : float, default=0.7
        Marker opacity (0-1).
    output_file : str, optional
        Path to save HTML file. If None, doesn't save.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    fig : plotly Figure
        The generated figure.
    
    Examples
    --------
    >>> # 2D shape
    >>> Y, labels = ShapeBuilder.distinct_points(['A', 'B'], [1, 2, 3])
    >>> plot_shape(Y, labels, color_by='a')
    
    >>> # 3D shape
    >>> Y, labels = ShapeBuilder.cylinder(np.arange(2020, 2025), np.arange(1, 13))
    >>> plot_shape(Y, labels, color_by='angle', output_file='cylinder.html')
    
    >>> # Hierarchical shape
    >>> Y, labels = ShapeBuilder.hierarchical_clusters(['A', 'B'], [1, 2, 3])
    >>> plot_shape(Y, labels, color_by='level_1', title='Nested Clusters')
    """
    
    # Validate input
    if Y.shape[0] != len(labels_df):
        raise ValueError(f"Y has {Y.shape[0]} points but labels_df has {len(labels_df)} rows")
    
    n_dims = Y.shape[1]
    if n_dims not in [2, 3]:
        raise ValueError(f"Y must be 2D or 3D, got shape {Y.shape}")
    
    # Determine color column
    if color_by is None:
        color_by = labels_df.columns[0]
    elif color_by not in labels_df.columns:
        raise ValueError(f"color_by='{color_by}' not found in labels. Available: {list(labels_df.columns)}")
    
    # Extract coordinates
    x, y = Y[:, 0], Y[:, 1]
    z = Y[:, 2] if n_dims == 3 else None
    
    # Prepare color data
    color_data = labels_df[color_by].values
    
    # Convert categorical to numeric for colorscale
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
    
    # Build hover text with all labels
    hover_parts = []
    for col in labels_df.columns:
        hover_parts.append(f"{col}=%{{customdata[{list(labels_df.columns).index(col)}]}}")
    hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"
    
    customdata = labels_df.values
    
    # Create trace
    if n_dims == 3:
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=dict(
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
            ),
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
    else:  # 2D
        trace = go.Scatter(
            x=x, y=y,
            mode="markers",
            customdata=customdata,
            hovertemplate=hovertemplate,
            marker=dict(
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
            ),
            showlegend=False,
        )
        
        fig = go.Figure(data=[trace])
        fig.update_xaxes(title="X", scaleanchor="y", scaleratio=1)
        fig.update_yaxes(title="Y")
    
    # Set title
    if title is None:
        label_cols = [col for col in labels_df.columns if col != 'pair']
        title = f"Shape: {' × '.join(label_cols)}"
    
    fig.update_layout(
        title=title,
        width=800,
        height=700,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    
    # Save if requested
    if output_file:
        fig.write_html(output_file, include_plotlyjs="cdn")
        print(f"Saved plot to {output_file}")
    
    # Show if requested
    if show:
        fig.show()
    
    return fig