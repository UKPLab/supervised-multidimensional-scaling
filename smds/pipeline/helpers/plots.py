import os
from typing import List, Union

import matplotlib
from sklearn.model_selection import KFold  # type: ignore[import-untyped]
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import gridspec

from smds import ComputedSMDSParametrization, SupervisedMDS, UserProvidedSMDSParametrization
from smds.pipeline.helpers.styling import (
    COL_CONTINUOUS,
    COL_DEFAULT,
    COL_DISCRETE,
    COL_SPATIAL,
    COL_USER_PROVIDED,
    get_shape_color,
)
from smds.shapes.base_shape import BaseShape
from smds.stress.stress_metrics import StressMetrics

matplotlib.use("Agg")


def create_plots(
    X: np.ndarray,
    y: np.ndarray,
    results_df: pd.DataFrame,
    shapes: List[Union[BaseShape, UserProvidedSMDSParametrization]],
    csv_path: str,
    experiment_name: str,
    n_folds: int = 5,
    smds_components: int = 2,
) -> None:
    """
    Creates a combined visualization:
    1. Scatter plot of the best manifold projection (left; out-of-sample when n_folds >= 2).
    2. Grid of bar charts showing rankings for ALL computed metrics (right).
    """
    sns.set_theme(style="whitegrid")

    valid_df = results_df[results_df["error"].isna()].copy()

    if valid_df.empty:
        print("No valid results to visualize.")
        return

    valid_df["display_name"] = valid_df["shape"].apply(lambda x: x.replace("Shape", ""))

    # Palette dictionary
    unique_shapes = valid_df["shape"].unique()
    custom_palette = {}

    for shape_name in unique_shapes:
        display_name = shape_name.replace("Shape", "")
        custom_palette[display_name] = get_shape_color(shape_name)

    # Identify metrics
    primary_metric = StressMetrics.SCALE_NORMALIZED_STRESS.value
    primary_col = f"mean_{primary_metric}"

    metric_cols = [c for c in valid_df.columns if c.startswith("mean_")]

    # Sort metrics: Primary first, then others
    if primary_col in metric_cols:
        metric_cols.remove(primary_col)
        metric_cols.insert(0, primary_col)

    if not metric_cols:
        print("No metric columns found to visualize.")
        return

    # Grid Layout
    n_metrics = len(metric_cols)
    n_rows = (n_metrics + 1) // 2

    # Calculate Figure Size
    fig_height = 5 * n_rows
    fig_width = 24

    fig = plt.figure(figsize=(fig_width, fig_height))

    fig.suptitle(f"Experiment: {experiment_name.replace('_', ' ')}", fontsize=24, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(n_rows, 4, width_ratios=[1, 1.5, 1, 1.5], wspace=0.4, hspace=0.6)

    shape_dict: dict[str, Union[BaseShape, UserProvidedSMDSParametrization]] = {}
    for s in shapes:
        shape_dict[s.__class__.__name__] = s
        name = getattr(s, "name", None)
        if name:
            shape_dict[name] = s

    # Metric Grid
    for idx, col_name in enumerate(metric_cols):
        metric_key = col_name.replace("mean_", "")
        display_title = metric_key.replace("_", " ").title()

        # Sort:
        local_df = valid_df.sort_values(by=col_name, ascending=False)

        best_row = local_df.iloc[0]
        best_shape_name = best_row["shape"]
        best_score_val = best_row[col_name]

        row = idx // 2
        col_offset = (idx % 2) * 2

        cell = gs[row, col_offset]
        best_shape_obj = shape_dict.get(best_shape_name)

        if best_shape_obj:
            if isinstance(best_shape_obj, BaseShape):
                parametrization = ComputedSMDSParametrization(n_components=smds_components, manifold=best_shape_obj)
            else:
                parametrization = best_shape_obj
            try:
                n_comp = getattr(parametrization, "n_components", smds_components)
                if n_folds >= 2:
                    kf = KFold(n_splits=n_folds, shuffle=False)
                    X_embedded = np.full((X.shape[0], n_comp), np.nan, dtype=np.float64)
                    for train_idx, test_idx in kf.split(X):
                        X_embedded[test_idx] = (
                            SupervisedMDS(parametrization)
                            .fit(X[train_idx], y[train_idx])
                            .transform(X[test_idx])
                        )
                else:
                    X_embedded = SupervisedMDS(parametrization).fit_transform(X, y)
                hue_vals = y[:, 0] if y.ndim > 1 else y
                is_discrete = len(np.unique(hue_vals)) < 20
                palette = "tab10" if is_discrete else "viridis"
                title_str = f"{display_title}\nBest: {best_shape_name} ({best_score_val:.4f})"
                if n_comp > 2:
                    title_str += f" [{n_comp}D]"

                if n_comp == 1:
                    ax_scatter = fig.add_subplot(cell)
                    sns.scatterplot(
                        x=X_embedded[:, 0],
                        y=np.zeros(X_embedded.shape[0]),
                        hue=hue_vals,
                        palette=palette,
                        alpha=0.8,
                        s=80,
                        ax=ax_scatter,
                        legend="full" if is_discrete else False,
                    )
                    ax_scatter.set_xlabel("Dim 1")
                    ax_scatter.set_ylabel("")
                    ax_scatter.set_title(title_str, fontsize=14, fontweight="bold")
                    sns.despine(ax=ax_scatter)
                elif n_comp == 2:
                    ax_scatter = fig.add_subplot(cell)
                    sns.scatterplot(
                        x=X_embedded[:, 0],
                        y=X_embedded[:, 1],
                        hue=hue_vals,
                        palette=palette,
                        alpha=0.8,
                        s=80,
                        edgecolor="w",
                        linewidth=0.5,
                        ax=ax_scatter,
                        legend="full" if is_discrete else False,
                    )
                    ax_scatter.set_aspect("equal", adjustable="datalim")
                    ax_scatter.set_xlabel("Dim 1")
                    ax_scatter.set_ylabel("Dim 2")
                    ax_scatter.set_title(title_str, fontsize=14, fontweight="bold")
                    sns.despine(ax=ax_scatter)
                elif n_comp == 3:
                    ax_3d = fig.add_subplot(cell, projection="3d")
                    cmap_name = "tab10" if is_discrete else "viridis"
                    sc = ax_3d.scatter(
                        X_embedded[:, 0],
                        X_embedded[:, 1],
                        X_embedded[:, 2],
                        c=hue_vals,
                        cmap=cmap_name,
                        alpha=0.8,
                        s=50,
                    )
                    if is_discrete:
                        ax_3d.legend(*sc.legend_elements(), loc="upper right", fontsize=6)
                    ax_3d.set_xlabel("Dim 1")
                    ax_3d.set_ylabel("Dim 2")
                    ax_3d.set_zlabel("Dim 3")
                    ax_3d.set_title(title_str, fontsize=14, fontweight="bold")
                else:
                    sub_gs = cell.subgridspec(2, 2, hspace=0.35, wspace=0.35)
                    pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
                    for k, (i, j) in enumerate(pairs[:4]):
                        ax_sub = fig.add_subplot(sub_gs[k // 2, k % 2])
                        sns.scatterplot(
                            x=X_embedded[:, i],
                            y=X_embedded[:, j],
                            hue=hue_vals,
                            palette=palette,
                            alpha=0.8,
                            s=40,
                            ax=ax_sub,
                            legend=False,
                        )
                        ax_sub.set_xlabel(f"Dim {i + 1}")
                        ax_sub.set_ylabel(f"Dim {j + 1}")
                        ax_sub.set_aspect("equal", adjustable="datalim")
                        sns.despine(ax=ax_sub)
                        if k == 0:
                            ax_sub.set_title(title_str, fontsize=14, fontweight="bold")

            except Exception as e:
                ax_fail = fig.add_subplot(cell)
                ax_fail.text(0.5, 0.5, f"Error plotting: {e}", ha="center", va="center")
        else:
            ax_fail = fig.add_subplot(cell)
            ax_fail.text(0.5, 0.5, "Best shape object missing", ha="center", va="center")

        ax_bar = fig.add_subplot(gs[row, col_offset + 1])

        std_col = f"std_{metric_key}"
        has_std = std_col in local_df.columns

        sns.barplot(
            data=local_df,
            x=col_name,
            y="display_name",
            palette=custom_palette,
            ax=ax_bar,
            orient="h",
            hue="display_name",
            legend=False,
        )

        if has_std:
            ax_bar.errorbar(
                x=local_df[col_name],
                y=np.arange(len(local_df)),
                xerr=local_df[std_col],
                fmt="none",
                c="black",
                capsize=3,
                linewidth=1,
            )

        ax_bar.set_title(f"{display_title} - Ranking", fontsize=14, fontweight="bold")
        ax_bar.set_xlabel("Score (Higher is Better)")
        ax_bar.set_ylabel("")

        # Add labels if not too crowded
        if len(local_df) < 25:
            for i, (score, std) in enumerate(
                zip(local_df[col_name], local_df[std_col] if has_std else [0] * len(local_df))
            ):
                if pd.isna(score):
                    continue

                label = f"{score:.3f}"
                # place text inside or outside based on value
                ax_bar.text(
                    0.02,
                    i,
                    label,
                    color="white",
                    va="center",
                    fontweight="bold",
                    fontsize=8,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="black")],
                )

    # Add Legend (Bottom Right)
    category_map = {
        COL_CONTINUOUS: "Continuous Shape",
        COL_DISCRETE: "Discrete Shape",
        COL_SPATIAL: "Spatial Shape",
        COL_USER_PROVIDED: "User Provided",
        COL_DEFAULT: "Other",
    }

    # Identify which colors are actually used
    used_hex_codes = set(custom_palette.values())

    legend_handles = []
    for color_code, label in category_map.items():
        if color_code in used_hex_codes:
            patch = mpatches.Patch(color=color_code, label=label)
            legend_handles.append(patch)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.01),
            title="Shape Categories",
            title_fontsize=12,
            fontsize=10,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )

    # Save
    base_path = os.path.splitext(csv_path)[0]
    img_path = f"{base_path}_visualized.png"

    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual result saved under: {img_path}")
