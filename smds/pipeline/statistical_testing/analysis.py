import os
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def perform_st_analysis(
        aggregated_scores: pd.DataFrame,
        output_dir: str,
        metric_name: str
) -> None:
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
    os.makedirs(output_dir, exist_ok=True)

    #    This leaves a simple RangeIndex (0, 1, 2...) which scikit-posthocs treats as Block IDs.
    sanitized_df = aggregated_scores.dropna().reset_index(drop=True)

    if sanitized_df.empty:
        print("Error: No valid data for statistical testing (all rows contained NaNs).")
        return

    # Friedman Test
    # Transpose because friedmanchisquare expects samples as columns
    try:
        stat, p_value = stats.friedmanchisquare(*[sanitized_df[col] for col in sanitized_df.columns])
    except ValueError as e:
        print(f"Friedman test failed (likely insufficient variance or samples): {e}")
        return

    summary = {
        "test": "Friedman Chi-Square",
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

    # Save simple text summary
    with open(os.path.join(output_dir, "friedman_summary.txt"), "w") as f:
        f.write(str(summary))

    if p_value >= 0.05:
        print(f"Friedman test not significant (p={p_value:.4f}). Skipping post-hoc.")
        return

    # Nemenyi Post-hoc Test
    p_values_matrix = sp.posthoc_nemenyi_friedman(sanitized_df)

    # Save P-values CSV
    p_values_matrix.to_csv(os.path.join(output_dir, "p_values.csv"))

    # Generate Heatmap
    plt.figure(figsize=(10, 8))
    sp.sign_plot(p_values_matrix)
    plt.suptitle(f"Nemenyi Test P-Values ({metric_name})", y=0.98, fontsize=14)
    plt.savefig(os.path.join(output_dir, "p_values_heatmap.png"), bbox_inches='tight')
    plt.close()

    # Critical Difference Diagram
    try:
        # Calculate average ranks on the Wide DataFrame
        ranks = sanitized_df.rank(axis=1, ascending=False).mean()

        plt.figure(figsize=(10, 4), dpi=150)

        sp.critical_difference_diagram(ranks, p_values_matrix)

        plt.title(f"Critical Difference Diagram ({metric_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cd_diagram.png"), bbox_inches="tight")
        plt.close()
    except AttributeError:
        print("Warning: scikit-posthocs version too old for critical_difference_diagram.")
    except Exception as e:
        print(f"Warning: Could not generate CD diagram: {e}")