#!/usr/bin/env python3
# filepath: plot_random_full_kl.py

import torch
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_model
import json
import argparse

# Path setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports from the project
from data.tokenizers import ASCIITokenizer

from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT
from models.gpt import GPT

from circuits import Circuit
from circuits.search.ablation import ZeroAblator
from circuits.search.divergence import compute_downstream_magnitudes, get_predicted_logits_from_full_circuit
from circuits.search.edges import compute_downstream_magnitudes_from_edges

from xavier.utils import compute_kl_divergence, create_random_edges, randomly_select_edges

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot KL divergence between random and full circuits')
    parser.add_argument('--layer', type=int, default=0, help='Layer number to analyze')
    parser.add_argument('--output', type=str, default='kl_divergence_plot.png', help='Output file name for plot')
    return parser.parse_args()

def load_model_and_weights():
    """Load the SparsifiedGPT model and weights"""
    device = torch.device("cpu")
    checkpoint_dir = Path(project_root) / "checkpoints"
    gpt_dir = checkpoint_dir / "shakespeare_64x4"
    sae_dir = checkpoint_dir / "standard.shakespeare_64x4"
    
    # Load GPT model
    gpt = GPT.load(gpt_dir, device=device)
    
    # Load SAE config
    sae_config_dir = sae_dir / "sae.json"
    with open(sae_config_dir, "r") as f:
        meta = json.load(f)
    config = SAEConfig(**meta)
    config.gpt_config = gpt.config
    
    # Create model using saved config
    model = SparsifiedGPT(config)
    model.gpt = gpt
    
    # Load SAE weights
    for layer_name, module in model.saes.items():
        weights_path = os.path.join(sae_dir, f"sae.{layer_name}.safetensors")
        load_model(module, weights_path, device=device.type)
        
    return model, device

def main():
    """
    Plot KL divergence between random and full circuits
    """
    args = parse_arguments()
    
    print("Loading model and weights...")
    model, device = load_model_and_weights()
    
    print("Setting up computation parameters...")
    ablator = ZeroAblator()
    layer_num = args.layer
    target_token_idx = 0
    num_downstream_features = model.config.n_features[layer_num]
    num_upstream_features = model.config.n_features[layer_num+1]
    num_samples = 2
    
    # Prompt for input
    prompt = "And let us once again assail your ea"
    tokenizer = ASCIITokenizer()
    input_ids = torch.tensor([tokenizer.encode(text=prompt)])
    assert input_ids.shape[1] <= model.config.gpt_config.block_size, "Input sequence length exceeds block size"

    print(f"Computing upstream & downstream magnitudes (full circuit)...")
    with model.use_saes(layers_to_patch=[layer_num, layer_num + 1]) as encoder_outputs:
        _ = model(input_ids)
        upstream_magnitudes = encoder_outputs[layer_num].feature_magnitudes
        downstream_magnitudes = encoder_outputs[layer_num + 1].feature_magnitudes
    
    # Get full circuit logits
    logits_full = get_predicted_logits_from_full_circuit(
        model,
        layer_num + 1,
        downstream_magnitudes,
        target_token_idx
    )   
    
    # Create random edges and compute KL divergences
    min_edges = 10
    max_edges = num_downstream_features*num_upstream_features
    num_circuits = 5
    num_edges_array = np.flip(np.linspace(min_edges, max_edges, num_circuits, dtype=int))
    kl_divergences = []

    # Initialize edge set to full circuit
    random_edges = create_random_edges(
            layer_l=layer_num, 
            num_features_l=model.config.n_features[layer_num], 
            num_features_l_plus_1=model.config.n_features[layer_num+1] , 
            num_edges=model.config.n_features[layer_num]*model.config.n_features[layer_num+1]
        )
    
    for num_edges in num_edges_array:
        print(f"Computing for {num_edges} random edges...")
    
        # Update random edges
        random_edges = randomly_select_edges(random_edges, num_edges)
        
        # Save time if computing full circuit
        if num_edges == num_downstream_features*num_upstream_features:

            logits_random = get_predicted_logits_from_full_circuit(
                model,
                layer_num + 1,
                downstream_magnitudes,
                target_token_idx
            )

        else:
            # Compute downstream magnitudes from random edges
            random_downstream_magnitudes = compute_downstream_magnitudes_from_edges(
                model,
                ablator,
                random_edges,
                upstream_magnitudes.squeeze(0),
                target_token_idx,
                num_samples
            )
            
            # Get random circuit logits
            logits_random = get_predicted_logits_from_full_circuit(
                model,
                layer_num + 1,
                random_downstream_magnitudes.unsqueeze(0),
                target_token_idx
            )
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(logits_full, logits_random)
        kl_divergences.append(kl_div)
        print(f"KL divergence with {num_edges} edges: {kl_div:.4f}")
    
    # Plot KL divergence
    plt.figure(figsize=(10, 6))
    plt.plot(num_edges_array.tolist(), kl_divergences, marker='o')
    plt.xlabel('num_edges')
    plt.ylabel('KL_div')
    plt.ylim(-1, 8)
    plt.title(f'KL Divergence Between Full Model and Random Circuit Between Layers {layer_num} & {layer_num+1}')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=num_downstream_features*num_upstream_features, color='r', linestyle='--', alpha=0.3)
    plt.savefig(args.output)
    plt.close()
    
    print(f"Plot saved as {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()