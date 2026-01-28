import os
import random
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from smds import SupervisedMDS
from smds.shapes.continuous_shapes import EuclideanShape, CircularShape
from smds.demos.family_tree.data_generation import generate_family_tree_data, load_names

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def find_last_token_idx(tokenizer, text, target):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoding["offset_mapping"]

    target_start = text.rfind(target)
    if target_start == -1:
        return -1
    target_end = target_start + len(target)

    matched_idx = -1
    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end == 0: continue

        if not (tok_end <= target_start or tok_start >= target_end):
            matched_idx = idx
            pass

    return matched_idx


def get_activations(df, model, tokenizer, layer=-1):
    model.eval()
    activations = []
    distances = []

    print("Recording activations...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        target_map = row['target_map']

        input_ids = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**input_ids, output_hidden_states=True)

        hidden_state = outputs.hidden_states[layer].squeeze(0).cpu().numpy()

        for name, dist in target_map.items():
            idx = find_last_token_idx(tokenizer, text, name)
            if idx != -1 and idx < len(hidden_state):
                vect = hidden_state[idx]
                activations.append(vect)
                distances.append(dist)
            else:
                pass

    return np.array(activations), np.array(distances)


def main():
    print("Setting up experiment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Generating data...")
    all_names = load_names()
    df = generate_family_tree_data(all_names, n_samples=50)
    print(f"Generated {len(df)} family trees.")
    print(f"Sample text: {df.iloc[0]['text']}")

    X, y = get_activations(df, model, tokenizer, layer=-1)

    print(f"Collected data shape: X={X.shape}, y={y.shape}")

    print("\nRunning Manifold Discovery Pipeline...")

    from smds.pipeline.discovery_pipeline import discover_manifolds

    y = y.astype(np.float64)
    X = X.astype(np.float64)

    results_df, save_path = discover_manifolds(
        X, y,
        experiment_name="family_tree_experiment",
        model_name="qwen",
        n_folds=5,
        n_jobs=-1,
        save_results=True,
        create_visualization=True
    )

    print("\nPipeline Results:")
    print(results_df[['shape', 'mean_scale_normalized_stress', 'std_scale_normalized_stress', 'error']])

    if not results_df.empty:
        winner = results_df.iloc[0]
        print(f"\nWinner: {winner['shape']} (Mean Score: {winner['mean_scale_normalized_stress']:.4f})")

    print("\nRunning Hierarchical Analysis...")
    from smds.shapes.discrete_shapes.hierarchical import HierarchicalShape

    y_hier = np.zeros((len(y), 2), dtype=np.float64)
    y_hier[:, 1] = y

    hierarchical_shape = HierarchicalShape(level_distances=[5.0, 1.0])

    results_hier, _ = discover_manifolds(
        X, y_hier,
        shapes=[hierarchical_shape],
        save_path=save_path,  # Append to existing results
        n_folds=5,
        n_jobs=-1,
        save_results=True,
        create_visualization=True
    )

    print("\nHierarchical Results:")
    print(results_hier[['shape', 'mean_scale_normalized_stress', 'std_scale_normalized_stress', 'error']])


if __name__ == "__main__":
    main()
