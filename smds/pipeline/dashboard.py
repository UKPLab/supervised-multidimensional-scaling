"""
Streamlit Dashboard for visualizing Manifold Discovery results.
Run this file via the wrapper `smds/pipeline/open_dashboard.py`
or directly via `uv run streamlit run smds/pipeline/dashboard.py`.
"""

import json
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st
import streamlit.components.v1 as components

from smds.pipeline.helpers.styling import COL_CONTINUOUS, COL_DEFAULT, COL_DISCRETE, COL_SPATIAL, SHAPE_COLORS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "saved_results")
ST_RESULTS_BASE = os.path.join(CURRENT_DIR, "statistical_testing", "st_results")


def load_data_with_metadata(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads CSV data and looks for a sibling 'metadata.json' file.
    """
    metadata = {}
    folder_path = os.path.dirname(file_path)
    meta_path = os.path.join(folder_path, "metadata.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata.json: {e}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(), {}

    return df, metadata


def create_p_value_heatmap(df: pd.DataFrame, theme: str) -> go.Figure:
    """
    Creates a discrete p-value heatmap with masked diagonal and custom buckets.
    """
    # 1. Mask diagonal logic
    df_masked = df.copy()
    np.fill_diagonal(df_masked.values, np.nan)

    # 2. Bucketize values for Discrete Coloring
    # -1 = Diagonal (Explicit White)
    # 3 = p < 0.001 (Highly Sig)
    # 2 = p < 0.01
    # 1 = p < 0.05
    # 0 = NS
    def get_bucket(x, r, c):
        if r == c: return -1  # Diagonal
        if np.isnan(x): return 0  # Treat missing as NS or handle separately
        if x < 0.001: return 3
        if x < 0.01: return 2
        if x < 0.05: return 1
        return 0

    # Apply logic element-wise using indices
    z_values = pd.DataFrame(
        [[get_bucket(df.iloc[r, c], r, c) for c in range(df.shape[1])] for r in range(df.shape[0])],
        index=df.index, columns=df.columns
    )

    # 3. Define Color Themes
    # Format: [NS, <0.05, <0.01, <0.001]
    themes = {
        "Reference Green": ['#fde0dd', '#a1d99b', '#31a354', '#006d2c'],
        "Scientific Blue": ['#f7fbff', '#bdd7e7', '#6baed6', '#2171b5'],
        "Royal Purple": ['#fcfbfd', '#dadaeb', '#9e9ac8', '#54278f'],  # New elegant option
    }

    # Get theme colors
    c = themes.get(theme, themes["Scientific Blue"])
    diagonal_color = 'white'

    # 4. Construct Discrete Colorscale
    # We have 5 integers: -1, 0, 1, 2, 3.
    # We map them to: [Diag, NS, Sig1, Sig2, Sig3]
    # In Plotly colorscales [0, 1], we slice into 5 chunks.
    # 0.0 - 0.2: Diagonal
    # 0.2 - 0.4: NS
    # 0.4 - 0.6: <0.05
    # 0.6 - 0.8: <0.01
    # 0.8 - 1.0: <0.001

    colorscale = [
        [0.0, diagonal_color], [0.2, diagonal_color],  # -1
        [0.2, c[0]], [0.4, c[0]],  # 0
        [0.4, c[1]], [0.6, c[1]],  # 1
        [0.6, c[2]], [0.8, c[2]],  # 2
        [0.8, c[3]], [1.0, c[3]],  # 3
    ]

    # Create text for hover (Original P-Values)
    # Set diagonal text to empty string
    text_values = df.map(lambda x: f"{x:.3f}" if not np.isnan(x) else "")
    for i in range(len(df)):
        text_values.iloc[i, i] = ""

    # 5. Construct Plot
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=df.columns,
        y=df.index,
        text=text_values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo='x+y+text',
        colorscale=colorscale,
        zmin=-1,
        zmax=3,
        showscale=True,
        colorbar=dict(
            tickmode="array",
            # Center ticks in the 5 blocks:
            # -1 block is at -0.6ish... actually for zmin=-1 zmax=3 range is 4.
            # let's manually place ticks relative to values -1, 0, 1, 2, 3
            tickvals=[0, 1, 2, 3],
            ticktext=["NS", "p < 0.05", "p < 0.01", "p < 0.001"],
            title="Significance"
        )
    ))

    fig.update_layout(
        title="Pairwise Nemenyi Test P-Values",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="SMDS Dashboard", layout="wide")

    # Custom CSS to reduce top padding
    st.markdown(
        """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧩 Manifold Discovery Dashboard")

    # Handle command-line argument for auto-selecting a file
    preselected_file = None
    if len(sys.argv) > 1:
        possible_file = sys.argv[1]
        preselected_file = os.path.basename(possible_file)

    if not os.path.exists(RESULTS_DIR):
        st.error(f"Results directory not found: {RESULTS_DIR}")
        return

    # Scan for experiment directories
    experiments = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    experiments.sort(reverse=True)

    # Logic to handle pre-selection
    default_index = 0
    if preselected_file in experiments:
        default_index = experiments.index(preselected_file)

    selected_exp_dir = st.sidebar.selectbox(
        "Select Experiment",
        experiments,
        index=default_index,  # Auto-select the specific experiment folder
    )

    if selected_exp_dir:
        exp_path = os.path.join(RESULTS_DIR, selected_exp_dir)

        csv_files = [f for f in os.listdir(exp_path) if f.endswith(".csv")]

        if not csv_files:
            st.error(f"No CSV results found in {selected_exp_dir}")
            return

        file_path = os.path.join(exp_path, csv_files[0])
        df, metadata = load_data_with_metadata(file_path)

        # Attempt to parse metadata from directory name
        display_name = selected_exp_dir
        try:
            name_parts = selected_exp_dir.split("_")
            if len(name_parts) >= 4:
                time_str = name_parts[-2]
                date_str = name_parts[-3]
                exp_name = "_".join(name_parts[:-3])

                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                display_name = f"🧪 **{exp_name}** | 📅 {date_str} {formatted_time} | 📂 `{selected_exp_dir}`"
        except Exception:
            pass

        st.markdown(display_name)

        if df.empty:
            st.error("⚠️ This CSV file is empty. No results to display.")
            return

        # Identify available metrics
        metric_cols = [c for c in df.columns if c.startswith("mean_")]

        if not metric_cols:
            st.error("⚠️ No metric columns found in the CSV. The file format might be incompatible.")
            st.write("Available columns:", df.columns.tolist())
            return

        # Layout
        # 1. Container for the "Winner" stats at the top
        top_stats_container = st.container()

        # 2. Main columns: Chart (Left) and Interactive Viz (Right)
        col_chart, col_viz = st.columns([1.2, 1])

        with col_chart:
            # Create a placeholder for the chart so it appears ABOVE the selector
            chart_placeholder = st.empty()

            # Default to Scale Normalized Stress if available, else first metric
            default_metric_ix = 0
            preferred_metric = "mean_scale_normalized_stress"
            if preferred_metric in metric_cols:
                default_metric_ix = metric_cols.index(preferred_metric)

            # Render the selector
            selected_metric = st.selectbox(
                "Select Metric to Visualize",
                metric_cols,
                index=default_metric_ix,
                format_func=lambda x: x.replace("mean_", "").replace("_", " ").title(),
            )

        # Sort results based on the selected metric
        df_sorted = df.sort_values(selected_metric, ascending=False).reset_index(drop=True)

        if len(df_sorted) == 0:
            st.error("⚠️ No valid results found in this file.")
            return

        best_shape = df_sorted.iloc[0]["shape"]
        best_score = df_sorted.iloc[0][selected_metric]

        with top_stats_container:
            c1, c2 = st.columns(2)
            c1.metric("Winner", best_shape)
            c2.metric("Best Score", f"{best_score:.4f}")

        df_sorted["display_name"] = df_sorted["shape"].apply(lambda x: x.replace("Shape", ""))

        std_col = selected_metric.replace("mean_", "std_")

        # Categorical Coloring Logic
        category_map = {
            COL_CONTINUOUS: "Continuous",
            COL_DISCRETE: "Discrete",
            COL_SPATIAL: "Spatial",
            COL_DEFAULT: "Other",
        }

        def get_category(shape_name: str) -> str:
            hex_color = SHAPE_COLORS.get(shape_name, COL_DEFAULT)
            return category_map.get(hex_color, "Other")

        df_sorted["category"] = df_sorted["shape"].apply(get_category)

        # Map categories to colors
        cat_to_hex = {v: k for k, v in category_map.items()}
        colors = df_sorted["category"].map(cat_to_hex).fillna(COL_DEFAULT).tolist()

        # Create Plotly Bar Chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df_sorted[selected_metric],
                y=df_sorted["display_name"],
                orientation="h",
                error_x=dict(
                    type="data",
                    array=df_sorted[std_col] if std_col in df_sorted.columns else None,
                    visible=True if std_col in df_sorted.columns else False,
                    color="white",
                    thickness=1.5,
                    width=3,
                ),
                text=df_sorted[selected_metric].apply(lambda x: f"{x:.4f}"),
                textposition="auto",
                insidetextanchor="middle",
                marker=dict(color=colors),
                hovertemplate=(
                        "<b>%{y}</b><br>" +
                        "Score: %{x:.4f}<br>" +
                        "Category: %{customdata}<br>" +
                        "<i style='color:yellow'>Click to visualize</i>" +
                        "<extra></extra>"
                ),
                customdata=df_sorted["category"],
            )
        )

        fig.update_layout(
            title="Shape Hypothesis Ranking",
            xaxis_title=selected_metric.replace("mean_", "").replace("_", " ").title(),
            yaxis=dict(
                title="",
                autorange="reversed",  # Puts the winner at the top
            ),
            height=max(500, len(df_sorted) * 40),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )

        # Render Plotly Chart
        with chart_placeholder:
            event = st.plotly_chart(fig, key=f"bar_chart_{selected_metric}", on_select="rerun", width="stretch")

        # Determine which shape is selected
        selected_shape_row = df_sorted.iloc[0]  # Default to the winner (top row)
        event_any: Any = event

        if event_any and event_any.selection and event_any.selection.point_indices:
            idx = event_any.selection.point_indices[0]
            selected_shape_row = df_sorted.iloc[idx]

        with col_viz:
            st.subheader(f"{selected_shape_row['display_name']} Shape")

            # Check if plot path exists
            plot_rel_path = selected_shape_row.get("plot_path")

            if pd.isna(plot_rel_path) or not plot_rel_path:
                st.info("No interactive plot available for this shape.")
            else:
                full_plot_path = os.path.join(exp_path, plot_rel_path)

                if os.path.exists(full_plot_path):
                    with open(full_plot_path, "r", encoding="utf-8") as f:
                        html_content = f.read()

                    components.html(html_content, height=600, scrolling=False)
                else:
                    st.warning(f"Plot file missing: {plot_rel_path}")

        # === Statistical Validation ===
        st_run_id = metadata.get("st_run_id")

        if st_run_id:
            # Header + Theme Selector
            col_head, col_theme = st.columns([3, 1])
            with col_head:
                st.header("📊 Statistical Validation")
            with col_theme:
                heatmap_theme = st.selectbox("Heatmap Theme",
                                             ["Reference Green", "Scientific Blue", "Royal Purple"], index=0)

            st_path = os.path.join(ST_RESULTS_BASE, st_run_id)

            if not os.path.exists(st_path):
                st.warning(f"Metadata found ({st_run_id}), but results missing.")
            else:
                clean_metric_name = selected_metric.replace("mean_", "")

                # Load Summary
                summary_path = os.path.join(st_path, "st_summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        full_summary = json.load(f)
                    metric_stats = full_summary.get(clean_metric_name)

                    if metric_stats:
                        # Compact Stats Row
                        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                        c1.metric("Friedman Stat", f"{metric_stats.get('statistic', 0):.2f}")
                        p_val = metric_stats.get('p_value', 1.0)
                        is_sig = metric_stats.get('significant', False)
                        c2.metric("P-Value", f"{p_val:.2e}", delta="Significant" if is_sig else "Not Sig",
                                  delta_color="normal" if is_sig else "off")

                        if not is_sig:
                            st.info("Results not statistically significant.")
                        else:
                            # Layout: Heatmap (Left, Large) + CD Diagram (Right, Smaller)
                            metric_dir = os.path.join(st_path, clean_metric_name)
                            col_heat, col_cd = st.columns([1.3, 1])

                            # 1. Heatmap
                            with col_heat:
                                pval_file = os.path.join(metric_dir, "p_values.csv")
                                if os.path.exists(pval_file):
                                    p_df = pd.read_csv(pval_file, index_col=0)
                                    fig_heat = create_p_value_heatmap(p_df, heatmap_theme)
                                    st.plotly_chart(fig_heat, width="stretch")
                                else:
                                    st.warning("P-Values CSV missing.")

                            # 2. CD Diagram
                            with col_cd:
                                st.subheader("Ranking Analysis")
                                cd_file = os.path.join(metric_dir, "cd_diagram.png")
                                if os.path.exists(cd_file):
                                    st.image(cd_file, width="stretch")
                                    st.caption(
                                        "Lower rank (left) is better. Shapes connected by a thick bar "
                                        "are **statistically tied** (performed equally well).")
                                else:
                                    st.warning("CD Diagram missing.")
                    else:
                        st.info(f"No stats for {clean_metric_name}")
                else:
                    st.error("Summary file missing.")
        else:
            # Placeholder / Instructions
            st.header("📊 Statistical Validation")

            st.info(
                "**No statistical validation data found for this experiment.**\n\n"
                "This appears to be a standard single-pass run. "
            )

            with st.expander("How to enable Statistical Validation?"):
                st.markdown("""
                        Statistical validation requires running the pipeline multiple times with different random seeds 
                        to verify that the ranking of shapes is robust and significant. Therefore you need to run the 
                        Statistical Testing wrapper instead of the standard pipeline.
                        Find more details in the README.
                        """)
        # === Detailed Results ===
        st.markdown("---")
        st.subheader("Detailed Results")
        st.dataframe(df_sorted.style.highlight_max(axis=0, subset=[selected_metric]), width=1500)

if __name__ == "__main__":
    main()
