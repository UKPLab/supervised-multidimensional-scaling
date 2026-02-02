import os
import re
from typing import Optional

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
from numpy.typing import NDArray


def generate_interactive_plot(
    X_embedded: NDArray[np.float64],
    y: NDArray[np.float64],
    shape_name: str,
    save_dir: str,
    Y_ideal: Optional[NDArray[np.float64]] = None,
) -> str:
    """
    Generates an interactive Plotly scatter plot (2D or 3D) and saves it as an HTML file.

    Args:
        X_embedded: The low-dimensional embedding (n_samples, n_components).
                    Supports 2D or 3D data.
        y: The labels or target values (n_samples,).
        shape_name: The name of the shape hypothesis (used for filename).
        save_dir: The directory where the HTML file should be saved.
        Y_ideal: Optional (n_samples, n_components) ideal manifold coordinates to draw as a line.

    Returns
    -------
        The filename of the saved HTML plot (e.g., 'CircularShape.html').
    """
    n_components = X_embedded.shape[1]

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", shape_name)
    filename = f"{safe_name}.html"
    filepath = os.path.join(save_dir, filename)

    # Determine if y is discrete or continuous for coloring
    unique_labels = np.unique(y)
    is_discrete = len(unique_labels) < 20

    # Prepare DataFrame-like dict for Plotly
    plot_data = {"x": X_embedded[:, 0], "y": X_embedded[:, 1], "label": y}

    # Adjust formatting for discrete labels to ensure categorical legend
    if is_discrete:
        plot_data["label"] = y.astype(str)

    if n_components == 3:
        plot_data["z"] = X_embedded[:, 2]

        fig = px.scatter_3d(
            plot_data,
            x="x",
            y="y",
            z="z",
            color="label",
            color_continuous_scale="Viridis",
            color_discrete_sequence=px.colors.qualitative.T10,
        )
        # Enforce correct aspect ratio in 3D
        if Y_ideal is not None and Y_ideal.shape[1] >= 3:
            order = np.argsort(y.ravel())
            line_x = Y_ideal[order, 0]
            line_y = Y_ideal[order, 1]
            line_z = Y_ideal[order, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    name="Ideal shape",
                    line=dict(color="rgba(128,128,128,0.9)", width=4),
                )
            )
        fig.update_layout(
            scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3", aspectmode="data"),
            margin=dict(l=0, r=120, b=0, t=10),
            showlegend=True,
            legend=dict(orientation="v", x=1.02, xanchor="left"),
        )

    else:
        # Default to 2D
        fig = px.scatter(
            plot_data,
            x="x",
            y="y",
            color="label",
            color_continuous_scale="Viridis",
            color_discrete_sequence=px.colors.qualitative.T10,
        )
       
        # Enforce 1:1 Aspect Ratio
        if Y_ideal is not None and Y_ideal.shape[1] >= 2:
            order = np.argsort(y.ravel())
            line_x = Y_ideal[order, 0]
            line_y = Y_ideal[order, 1]
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="Ideal shape",
                    line=dict(color="rgba(128,128,128,0.9)", width=2),
                )
            )
        fig.update_layout(
            xaxis=dict(title="Dim 1"),
            yaxis=dict(title="Dim 2", scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, b=0, t=10),
        )

    # Save to disk
    fig.write_html(filepath, include_plotlyjs="cdn")

    return filename
