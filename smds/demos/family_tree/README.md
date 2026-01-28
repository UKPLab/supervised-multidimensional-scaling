# Familly Tree Experiment

Hypothesis: Concepts such as family trees may intrinsically take such organization in the latent space.

## Details

SMDS allows studying embeddings with a hierarchical nature. Concepts such as family trees may intrinsically take such
organization in the latent space. These are functional steps to validate this hypothesis:

A dataset of paragraphs describing a family tree is prepared. Unique names are sampled from a known set (see emailed
resources) and organized into a family tree. This family tree is then parsed into a text that describes it. E.g. "Anna's
parents are Sofia and Luke. Sofia's parents are Agnes and Robert. Luke's parents are George and Daniela." describes a
family tree from a child to its grandparents;
The paragraphs are fed to an LLM and activations in correspondence to the names tokens are recorded (see emailed
resources);
Manifold search is applied on these activations, using as features quantities like tree distance (1 for parent-child, 2
for grandparent-child). Several hypothesis manifolds are compared and one identified as the winner.

## Run

- Available models: gpt2, qwen, llama (see scripts).
- Data geeneration can be found in `data_generation.py`
- `smds/demos/resources/names.csv` contains unique names.
- `smds/demos/resources/sketch_record_activations.py` is a script for recording activations.

## Results

- **GPT-2:** 0.694 (Circular) vs. 0.749 (Hierarchical)
- **Llama:** 0.841 (Circular) vs. 0.878 (Hierarchical)
- **Qwen:** 0.787 (Circular) vs. 0.834 (Hierarchical)