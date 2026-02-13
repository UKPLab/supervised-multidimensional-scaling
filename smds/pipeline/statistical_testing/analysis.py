import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import scikit_posthocs as sp  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]


def perform_st_analysis(aggregated_scores: pd.DataFrame, output_dir: str, metric_name: str) -> Dict[str, Any]:
    """
    Performs Friedman test and Nemenyi post-hoc analysis on aggregated fold scores.
    Generates and saves:
    1. A Heatmap of p-values.
    2. A Critical Difference (CD) diagram.

    Args:
        aggregated_scores: DataFrame where index=(Run, Fold), Columns=Shapes.
        output_dir: Directory to save results.
        metric_name: Name of the metric used (for titles).
    """
    # Create subfolder for this specific metric
    metric_dir = os.path.join(output_dir, metric_name)
    os.makedirs(metric_dir, exist_ok=True)

    #    This leaves a simple RangeIndex (0, 1, 2...) which scikit-posthocs treats as Block IDs.
    sanitized_df = aggregated_scores.dropna().reset_index(drop=True)

    if sanitized_df.empty:
        print(f"[{metric_name}] Error: No valid data for statistical testing.")
        return {"error": "No valid data (NaNs)"}

    # Friedman Test
    # Transpose because friedmanchisquare expects samples as columns
    try:
        stat, p_value = stats.friedmanchisquare(*[sanitized_df[col] for col in sanitized_df.columns])
    except ValueError as e:
        print(f"[{metric_name}] Friedman test failed: {e}")
        return {"error": str(e)}

    summary = {
        "test": "Friedman Chi-Square",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }

    if p_value >= 0.05:
        print(f"[{metric_name}] Not significant (p={p_value:.4f}). Skipping post-hoc.")
        return summary

    # Nemenyi Post-hoc Test
    p_values_matrix = sp.posthoc_nemenyi_friedman(sanitized_df)

    # Save P-values CSV
    p_values_matrix.to_csv(os.path.join(metric_dir, "p_values.csv"))

    # Generate Heatmap
    plt.figure(figsize=(10, 8))
    sp.sign_plot(p_values_matrix)
    plt.suptitle(f"Nemenyi Test P-Values ({metric_name})", y=0.98, fontsize=14)
    plt.savefig(os.path.join(metric_dir, "p_values_heatmap.png"), bbox_inches="tight")
    plt.close()

    # Critical Difference Diagram
    try:
        ranks = sanitized_df.rank(axis=1, ascending=False).mean()

        plt.figure(figsize=(10, 4), dpi=150)
        sp.critical_difference_diagram(ranks, p_values_matrix)
        plt.suptitle(f"Critical Difference Diagram ({metric_name})", y=0.98, fontsize=14)

        plt.savefig(os.path.join(metric_dir, "cd_diagram.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[{metric_name}] Warning: Could not generate CD diagram: {e}")

    return summary
