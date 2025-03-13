#!/usr/bin/env python3
# filepath: plot_random_full_kl_batched.py

import torch
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_model
import json
import argparse
from safetensors.torch import load_file

# Path setup
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports from the project
from config.sae.models import SAEConfig
from models.sparsified import SparsifiedGPT
from models.gpt import GPT

from circuits import Circuit
from circuits.search.ablation import ZeroAblator
from circuits.search.divergence import get_batched_predicted_logits_from_full_circuit
from circuits.search.edges import compute_batched_downstream_magnitudes_from_edges

from xavier.utils import compute_kl_divergence, create_tokenless_edges_from_array, get_attribution_rankings
import time

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

    # Set up data parameters
    num_prompts = 1
    sequence_length = 128

    # Load validation data
    val_array = np.load('../data/shakespeare/val_000000.npy')
    val_tensor = torch.from_numpy(val_array).to(torch.long)  # Convert to long tensor

    # Calculate number of complete chunks we can make
    num_chunks = val_tensor.shape[0] // sequence_length
    usable_data = val_tensor[:num_chunks * sequence_length]
    reshaped_val_tensor = usable_data.reshape(num_chunks, sequence_length)
    input_ids = reshaped_val_tensor[:num_prompts, :].to(device)  # Also move to the correct device

    print("Setting up computation parameters...")
    ablator = ZeroAblator()
    layer_num = args.layer
    target_token_idx = sequence_length - 1
    num_downstream_features = model.config.n_features[layer_num]
    num_upstream_features = model.config.n_features[layer_num+1]
    num_samples = 2

    print(f"Computing upstream & downstream magnitudes (full circuit)...")
    with model.use_saes(layers_to_patch=[layer_num, layer_num + 1]) as encoder_outputs:
        _ = model(input_ids)
        upstream_magnitudes = encoder_outputs[layer_num].feature_magnitudes
        downstream_magnitudes = encoder_outputs[layer_num + 1].feature_magnitudes

    # Get full circuit logits
    logits_full = get_batched_predicted_logits_from_full_circuit(
        model,
        layer_num + 1,
        downstream_magnitudes.unsqueeze(1),
        target_token_idx
    )
    
    # Create random edges and compute KL divergences
    min_edges = 5
    max_edges = num_downstream_features*num_upstream_features
    num_circuits = 5
    num_edges_array = np.flip(np.linspace(min_edges, max_edges, num_circuits, dtype=int))

    # Selecting a few edge numbers for testing
    # num_edges_array = np.array([num_downstream_features * num_upstream_features - 1, ])
    # num_edges_array = np.array([num_downstream_features * num_upstream_features - 1])
    
    kl_divergences = []

    # Create random edge array 
    edge_arr = np.array([(a,b) for a in range(num_upstream_features) for b in range(num_downstream_features)])
    full_edge_arr = np.random.permutation(edge_arr)

    # # Load gradient attributions
    # file_path = "../Andy/data/attributions.safetensors"
    # tensors = load_file(file_path)

    # # Get attribution rankings
    # full_edge_arr, _ = get_attribution_rankings(tensors[f'attributions{layer_num}-{layer_num+1}'])

    
    for num_edges in num_edges_array:    
        # Update random edges
        random_edge_arr = full_edge_arr[:num_edges]
        random_edges = create_tokenless_edges_from_array(random_edge_arr, layer_num)

        print(f"Computing for {num_edges} random edges...")
        
        # Save time if computing full circuit
        if num_edges == num_downstream_features * num_upstream_features:

            logits_random = get_batched_predicted_logits_from_full_circuit(
                model,
                layer_num + 1,
                downstream_magnitudes.unsqueeze(1),
                target_token_idx
            )

        else:
            # Compute downstream magnitudes from random edges
            random_downstream_magnitudes = compute_batched_downstream_magnitudes_from_edges(
                model,
                ablator,
                random_edges,
                upstream_magnitudes,
                target_token_idx,
                num_samples
            )

            # Get random circuit logits
            logits_random = get_batched_predicted_logits_from_full_circuit(
                model,
                layer_num + 1,
                random_downstream_magnitudes.unsqueeze(1),
                target_token_idx
            )
        
        # Compute KL divergence
        kl_div = np.average([compute_kl_divergence(logits_full[i], logits_random[i]) for i in range(num_prompts)])
        kl_divergences.append(kl_div)
        print(f"KL divergence with {num_edges} edges: {kl_div:.4f}")
    
    # Plot KL divergence
    plt.figure(figsize=(10, 6))
    plt.plot(num_edges_array.tolist(), kl_divergences, marker='o')
    plt.xlabel('num_edges')
    plt.ylabel('KL_div')
    plt.ylim(-1, 10)
    plt.title(f'KL Divergence Between Full Model and Random Circuit Between Layers {layer_num} & {layer_num+1}')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=num_downstream_features*num_upstream_features, color='r', linestyle='--', alpha=0.3)
    plt.savefig(args.output)
    plt.close()
    
    print(f"Plot saved as {args.output}")
    print("Done!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")