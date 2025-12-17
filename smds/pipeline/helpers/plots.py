import os
from typing import List

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import gridspec

from smds import SupervisedMDS
from smds.shapes.base_shape import BaseShape

matplotlib.use("Agg")


def create_plots(
    X: np.ndarray,
    y: np.ndarray,
    results_df: pd.DataFrame,
    shapes: List[BaseShape],
    csv_path: str,
    experiment_name: str,
) -> None:
    """
    Creates a combined visualization:
    1. Scatter plot of the best manifold projection (left).
    2. Ranking bar chart of all tested shapes (right).
    """
    sns.set_theme(style="whitegrid")

    valid_df = results_df[results_df["mean_test_score"].notna()].copy()

    if valid_df.empty:
        print("No valid results to visualize.")
        return

    valid_df = valid_df.sort_values(by="mean_test_score", ascending=False)

    best_row = valid_df.iloc[0]
    best_shape_name = best_row["shape"]
    best_score = best_row["mean_test_score"]

    shape_dict = {s.__class__.__name__: s for s in shapes}
    best_shape_obj = shape_dict.get(best_shape_name)

    if not best_shape_obj:
        print(f"Warning: Best shape '{best_shape_name}' not found in the shape list.")
        return

    print(f"Generating plot for best shape: {best_shape_name} (Score: {best_score:.4f})...")
    estimator = SupervisedMDS(n_components=2, manifold=best_shape_obj)
    try:
        X_embedded = estimator.fit_transform(X, y)
    except Exception as e:
        print(f"Error while calculating the embedding for the plot: {e}")
        return

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    ax_scatter = plt.subplot(gs[0])

    unique_labels = np.unique(y)
    is_discrete = len(unique_labels) < 20

    palette = "tab10" if is_discrete else "viridis"

    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=y,
        palette=palette,
        alpha=0.8,
        s=60,
        edgecolor="w",
        linewidth=0.5,
        ax=ax_scatter,
        legend="full" if is_discrete else False,
    )

    ax_scatter.set_title(f"Best Manifold: {best_shape_name}\nScore: {best_score:.4f}", fontsize=14, fontweight="bold")
    ax_scatter.set_xlabel("Dim 1")
    ax_scatter.set_ylabel("Dim 2")

    sns.despine(ax=ax_scatter)

    ax_bars = plt.subplot(gs[1])

    valid_df["display_name"] = valid_df["shape"].apply(lambda x: x.replace("Shape", ""))

    sns.barplot(
        data=valid_df,
        x="mean_test_score",
        y="display_name",
        hue="display_name",
        palette="viridis",
        ax=ax_bars,
        orient="h",
        legend=False,
    )

    ax_bars.errorbar(
        x=valid_df["mean_test_score"],
        y=np.arange(len(valid_df)),
        xerr=valid_df["std_test_score"],
        fmt="none",
        c="black",
        capsize=3,
    )

    ax_bars.set_title(f"Manifold Leaderboard ({experiment_name})", fontsize=14, fontweight="bold")
    ax_bars.set_xlabel("Mean Test Score (CV)")
    ax_bars.set_ylabel("")

    for i, (score, std) in enumerate(zip(valid_df["mean_test_score"], valid_df["std_test_score"])):
        ax_bars.text(
            0.02,
            i,
            f"{score:.3f} ±{std:.3f}",
            color="white",
            va="center",
            fontweight="bold",
            fontsize=9,
            path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
        )

    sns.despine(ax=ax_bars, left=True, bottom=False)

    base_path = os.path.splitext(csv_path)[0]
    img_path = f"{base_path}_dashboard.png"

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Dashboard saved under: {img_path}")
