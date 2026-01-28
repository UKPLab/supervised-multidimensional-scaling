import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def get_latest_results(experiment_name, model_name, base_results_dir):
    model_dir = os.path.join(base_results_dir, experiment_name, model_name)
    if not os.path.exists(model_dir):
        print(f"No results found for {model_name} in {model_dir}")
        return None

    subdirs = glob.glob(os.path.join(model_dir, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    if not subdirs:
        return None

    def parse_timestamp(d):
        basename = os.path.basename(d)
        ts_str = "_".join(basename.split("_")[:2])
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d_%H-%M")
        except ValueError:
            return datetime.min

    latest_dir = max(subdirs, key=parse_timestamp)
    results_file = os.path.join(latest_dir, "results.csv")

    if os.path.exists(results_file):
        print(f"Found latest result for {model_name}: {results_file}")
        return pd.read_csv(results_file)
    return None


def plot_results(models, experiment_name="family_tree_experiment"):
    # Path to saved_results relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_results_dir = os.path.abspath(os.path.join(script_dir, "../../pipeline/saved_results"))

    combined_data = []

    for model in models:
        df = get_latest_results(experiment_name, model, base_results_dir)
        if df is not None:
            df['model'] = model
            combined_data.append(df)

    if not combined_data:
        print("No data found to plot.")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    metrics = [col for col in full_df.columns if col.startswith("mean_") and "stress" in col]

    if not metrics:
        print("No stress metrics found to plot.")
        return

    sns.set_context("talk")
    sns.set_style("white")

    conclusion_dir = os.path.join(base_results_dir, experiment_name, "conclusion")
    os.makedirs(conclusion_dir, exist_ok=True)
    print(f"Saving plots to {conclusion_dir}")

    for metric in metrics:
        plt.figure(figsize=(14, 8))

        pivot_df = full_df.pivot(index="model", columns="shape", values=metric)
        col_order = pivot_df.mean().sort_values().index
        pivot_df = pivot_df[col_order]
        _ = sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            linewidths=.5,
            cbar_kws={'label': 'Stress (Lower is Better)'}
        )

        clean_metric_name = metric.replace("mean_", "").replace("_", " ").title()
        plt.title(f"{clean_metric_name}\n({experiment_name})")
        plt.xlabel("Shape Hypothesis")
        plt.ylabel("Model")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_filename = f"heatmap_{metric}.png"
        output_path = os.path.join(conclusion_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        print(f"Saved {output_filename}")
        plt.close()


if __name__ == "__main__":
    models_to_compare = ["gpt2", "qwen", "llama"]
    plot_results(models_to_compare)
