#!/usr/bin/env python3
# filepath: compute_downstream_magnitudes.py

import torch
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_model
import json

# Path setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports from the project
from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT
from models.gpt import GPT

from circuits import Circuit, Edge, Node
from circuits.search.ablation import ZeroAblator
from circuits.search.edges import compute_downstream_magnitudes_from_edges
from circuits.search.divergence import get_predicted_logits_from_full_circuit

def main():
    """
    Compute downstream magnitudes from edges using a SparsifiedGPT model.
    This script replicates the functionality in demo_compute_downstream_magnitudes_from_edges.ipynb
    """
    print("Setting up environment...")
    # Loading SparsifiedGPT
    device = torch.device("cpu")
    checkpoint_dir = Path(project_root) / "checkpoints"
    gpt_dir = checkpoint_dir / "shakespeare_64x4"
    sae_dir = checkpoint_dir / "standard.shakespeare_64x4"
    
    # Load GPT model
    print("Loading GPT model...")
    gpt = GPT.load(gpt_dir, device=device)
    
    # Load SAE config
    print("Loading SAE configuration...")
    sae_config_dir = sae_dir / "sae.json"
    with open(sae_config_dir, "r") as f:
        meta = json.load(f)
    config = SAEConfig(**meta)
    config.gpt_config = gpt.config
    
    # Create model using saved config
    print("Creating SparsifiedGPT model...")
    model = SparsifiedGPT(config)
    model.gpt = gpt
    
    # Load SAE weights
    print("Loading SAE weights...")
    for layer_name, module in model.saes.items():
        weights_path = os.path.join(sae_dir, f"sae.{layer_name}.safetensors")
        load_model(module, weights_path, device=device.type)
    
    # Set up parameters for computation
    print("Setting up computation parameters...")
    ablator = ZeroAblator()
    num_samples = 2
    
    T, F = 5, 512
    upstream_magnitudes = torch.randn(T, F)
    original_downstream_magnitudes = torch.randn(T, F)
    target_token_idx = 0
    
    layer_num = 0
    
    # Define edges
    fixed_edges = frozenset([
        Edge(
            Node(layer_idx=0, token_idx=0, feature_idx=0),
            Node(layer_idx=1, token_idx=0, feature_idx=0)
        ),
        Edge(
            Node(layer_idx=0, token_idx=0, feature_idx=1),
            Node(layer_idx=1, token_idx=0, feature_idx=0)
        ),
        Edge(
            Node(layer_idx=0, token_idx=0, feature_idx=1),
            Node(layer_idx=1, token_idx=0, feature_idx=1)
        ),
        Edge(
            Node(layer_idx=0, token_idx=0, feature_idx=1),
            Node(layer_idx=1, token_idx=0, feature_idx=2)
        ),
        Edge(
            Node(layer_idx=0, token_idx=0, feature_idx=2),
            Node(layer_idx=1, token_idx=0, feature_idx=3)
        ),
    ])
    
    # Compute downstream magnitudes from edges
    print("Computing downstream magnitudes from edges...")
    computed_magnitudes = compute_downstream_magnitudes_from_edges(
        model,
        ablator,
        fixed_edges,
        upstream_magnitudes,
        target_token_idx,
        num_samples
    )
    
    # Get predicted logits from computed magnitudes
    print("Computing predicted logits...")
    logits_edges = get_predicted_logits_from_full_circuit(
        model,
        layer_num + 1,
        computed_magnitudes.unsqueeze(0),
        target_token_idx
    )
    
    # Display results
    print("\nResults:")
    print(f"Computed magnitudes shape: {computed_magnitudes.shape}")
    print(f"Logits shape: {logits_edges.shape}")
    
    print("Done!")

if __name__ == "__main__":
    main()