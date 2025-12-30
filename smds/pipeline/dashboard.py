"""
Streamlit Dashboard for visualizing Manifold Discovery results.
Run this file via the wrapper `smds/pipeline/open_dashboard.py`
or directly via `uv run streamlit run smds/pipeline/dashboard.py`.
"""

import os
import sys

import plotly.graph_objects as go
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
import streamlit.components.v1 as components

from smds.pipeline.helpers.styling import (
    SHAPE_COLORS,
    COL_DEFAULT,
    COL_CONTINUOUS,
    COL_DISCRETE,
    COL_SPATIAL
)

# Locate results directory relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "saved_results")


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


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
        df = load_data(file_path)

        # Attempt to parse metadata from directory name: {Experiment}_{Date}_{Time}_{UUID}
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
            # Fallback to filename if parsing structure doesn't match
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
                format_func=lambda x: x.replace("mean_", "").replace("_", " ").title()
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
            COL_DEFAULT: "Other"
        }

        def get_category(shape_name):
            hex_color = SHAPE_COLORS.get(shape_name, COL_DEFAULT)
            return category_map.get(hex_color, "Other")

        df_sorted["category"] = df_sorted["shape"].apply(get_category)

        # Map categories to colors
        cat_to_hex = {v: k for k, v in category_map.items()}
        colors = df_sorted["category"].map(cat_to_hex).fillna(COL_DEFAULT).tolist()

        # Create Plotly Bar Chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_sorted[selected_metric],
            y=df_sorted["display_name"],
            orientation='h',
            error_x=dict(
                type='data',
                array=df_sorted[std_col] if std_col in df_sorted.columns else None,
                visible=True if std_col in df_sorted.columns else False,
                color='white',
                thickness=1.5,
                width=3
            ),
            text=df_sorted[selected_metric].apply(lambda x: f"{x:.4f}"),
            textposition='auto',
            insidetextanchor='middle',
            marker=dict(color=colors),
            hovertemplate=(
                    "<b>%{y}</b><br>" +
                    "Score: %{x:.4f}<br>" +
                    "Category: %{customdata}<extra></extra>"
            ),
            customdata=df_sorted["category"]
        ))

        fig.update_layout(
            title="Shape Hypothesis Ranking",
            xaxis_title=selected_metric.replace("mean_", "").replace("_", " ").title(),
            yaxis=dict(
                title="",
                autorange="reversed"  # Puts the winner at the top
            ),
            height=max(500, len(df_sorted) * 40),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )

        # Render Plotly Chart into the Placeholder (Left Column, Top)
        with chart_placeholder:
            event = st.plotly_chart(
                fig,
                key=f"bar_chart_{selected_metric}",
                on_select="rerun",
                width="stretch"
            )

        # Determine which shape is selected
        selected_shape_row = df_sorted.iloc[0]  # Default to the winner (top row)
        # Handle Selection Event
        if event and event.selection and event.selection.point_indices:
            idx = event.selection.point_indices[0]
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
                    with open(full_plot_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    components.html(html_content, height=600, scrolling=False)
                else:
                    st.warning(f"Plot file missing: {plot_rel_path}")

        # Detailed Data Table
        st.subheader("Detailed Results")
        st.dataframe(
            df_sorted.style.highlight_max(axis=0, subset=[selected_metric]),
            width='stretch'
        )

        # Error Report
        if "error" in df_sorted.columns and df_sorted["error"].notna().any():
            st.warning("⚠️ Some shapes failed to run:")
            st.table(df_sorted[df_sorted["error"].notna()][["shape", "error"]])


if __name__ == "__main__":
    main()
