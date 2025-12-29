"""
Streamlit Dashboard for visualizing Manifold Discovery results.
Run this file via the wrapper `smds/pipeline/open_dashboard.py`
or directly via `uv run streamlit run smds/pipeline/dashboard.py`.
"""

import os
import sys

import altair as alt
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

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

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    files.sort(reverse=True)

    # Logic to handle pre-selection
    default_index = 0
    if preselected_file in files:
        default_index = files.index(preselected_file)

    selected_file = st.sidebar.selectbox(
        "Select Experiment",
        files,
        index=default_index,  # Auto-select the specific file!
    )

    if selected_file:
        file_path = os.path.join(RESULTS_DIR, selected_file)
        df = load_data(file_path)

        # Attempt to parse metadata from filename: {Experiment}_{Date}_{Time}_{UUID}.csv
        display_name = selected_file
        try:
            name_parts = selected_file.replace(".csv", "").split("_")
            if len(name_parts) >= 4:
                time_str = name_parts[-2]
                date_str = name_parts[-3]
                exp_name = "_".join(name_parts[:-3])

                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                display_name = f"🧪 **{exp_name}** | 📅 {date_str} {formatted_time} | 📄 `{selected_file}`"
        except Exception:
            # Fallback to filename if parsing structure doesn't match
            pass

        st.markdown(display_name)

        if df.empty:
            st.error("⚠️ This CSV file is empty. No results to display.")
            return

        # Identify available metrics
        metric_cols = [c for c in df.columns if c.startswith("mean_")]

        # Default to Scale Normalized Stress if available, else first metric
        default_metric_ix = 0
        preferred_metric = "mean_scale_normalized_stress"
        if preferred_metric in metric_cols:
            default_metric_ix = metric_cols.index(preferred_metric)

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

        col1, col2 = st.columns(2)
        col1.metric("Winner", best_shape)
        col2.metric("Best Score", f"{best_score:.4f}")

        df_sorted["display_name"] = df_sorted["shape"].apply(lambda x: x.replace("Shape", ""))

        # 2. Calculate Error Bars
        std_col = selected_metric.replace("mean_", "std_")
        if std_col in df_sorted.columns:
            df_sorted["lower"] = df_sorted[selected_metric] - df_sorted[std_col]
            df_sorted["upper"] = df_sorted[selected_metric] + df_sorted[std_col]
        else:
            df_sorted["lower"] = df_sorted[selected_metric]
            df_sorted["upper"] = df_sorted[selected_metric]

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

        present_categories = df_sorted["category"].unique()
        scale_domain = []
        scale_range = []

        for hex_code, cat_name in category_map.items():
            if cat_name in present_categories:
                scale_domain.append(cat_name)
                scale_range.append(hex_code)

        # Altair Chart
        sorted_names = df_sorted["display_name"].tolist()

        base = alt.Chart(df_sorted).encode(
            y=alt.Y(
                "display_name",
                sort=sorted_names,
                title="Shape Hypothesis"
            ),
        )

        # Bar Chart
        bars = base.mark_bar().encode(
            x=alt.X(selected_metric, title=selected_metric.replace("mean_", "").replace("_", " ").title()),
            color=alt.Color(
                "category",
                scale=alt.Scale(domain=scale_domain, range=scale_range),
                legend=alt.Legend(title="Shape Category")
            ),
            tooltip=["shape", "category", selected_metric, std_col, "params"] if std_col in df_sorted.columns else [
                "shape", selected_metric]
        )

        # Error Bar Line
        error_rule = base.mark_rule(color="white").encode(
            x="lower",
            x2="upper"
        )

        error_cap_lower = base.mark_tick(color="white", thickness=1.5, size=10).encode(
            x="lower"
        )
        error_cap_upper = base.mark_tick(color="white", thickness=1.5, size=10).encode(
            x="upper"
        )

        # Value Labels
        text_labels = base.mark_text(
            align="center",
            baseline="middle",
            color="white",
            fontWeight="bold"
        ).transform_calculate(
            mid_x=f"datum['{selected_metric}'] / 2"
        ).encode(
            x=alt.X("mid_x:Q"),
            text=alt.Text(selected_metric, format=".4f")
        )

        # Combine all layers
        final_chart = (bars + error_rule + error_cap_lower + error_cap_upper + text_labels).properties(
            height=max(400, len(df_sorted) * 30)
        )

        st.altair_chart(final_chart, theme="streamlit", width='stretch')

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
