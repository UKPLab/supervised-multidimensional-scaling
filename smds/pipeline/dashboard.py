"""
Streamlit Dashboard for visualizing Manifold Discovery results.
Run this file via the wrapper `smds/pipeline/open_dashboard.py`
or directly via `uv run streamlit run smds/pipeline/dashboard.py`.
"""

import os
import sys

import altair as alt  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-untyped]
import streamlit as st  # type: ignore[import-not-found]

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

        # Sort results to determine the winner
        df_sorted = df.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

        best_shape = df_sorted.iloc[0]["shape"]
        best_score = df_sorted.iloc[0]["mean_test_score"]

        col1, col2 = st.columns(2)
        col1.metric("Winner", best_shape)
        col2.metric("Best Score", f"{best_score:.4f}")

        # Comparison Chart
        chart = (
            alt.Chart(df_sorted)
            .mark_bar()
            .encode(
                x=alt.X("mean_test_score", title="Mean Score"),
                y=alt.Y("shape", sort="-x", title="Shape Hypothesis"),
                color=alt.Color("mean_test_score", scale={"scheme": "viridis"}),
                tooltip=["shape", "mean_test_score", "std_test_score", "params"],
            )
            .properties(height=400)
        )

        st.altair_chart(chart, theme="streamlit")

        # Detailed Data Table
        st.subheader("Detailed Results")
        st.dataframe(
            df_sorted.style.highlight_max(axis=0, subset=["mean_test_score"]),
        )

        # 5. Error Report
        if "error" in df_sorted.columns and df_sorted["error"].notna().any():
            st.warning("⚠️ Some shapes failed to run:")
            st.table(df_sorted[df_sorted["error"].notna()][["shape", "error"]])


if __name__ == "__main__":
    main()
