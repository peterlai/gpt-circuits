import argparse
from pathlib import Path
import torch
import datetime
from safetensors.torch import load_file
import sys
import re
import numpy as np

import matplotlib.pyplot as plt

# Path setup
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Importing custom modules
from xavier.utils import load_experiments_and_extract_data, bootstrap_ci

def main():

    parser = argparse.ArgumentParser(description="Process SAE variants")
    parser.add_argument("--run-index", type=str, default="testing", help="Index of the run")
    parser.add_argument('--sae-variants', nargs='+', help='List of SAE variants')
    parser.add_argument('--edge-selections', nargs='+', help='List of edge selection methods')
    parser.add_argument('--upstream-layers', nargs='+', help='List of upstream layers')
    
    args = parser.parse_args()

    run_idx = args.run_index
    sae_variants = args.sae_variants
    edge_selections = args.edge_selections
    layers = args.upstream_layers

    # DATA LOADING
    exp_output = 'kl_divergence'
    data_dir = Path.cwd() / 'xavier' / 'experiments' / 'data' / f'{run_idx}'

    # Infer sweep type
    if len(edge_selections)==1:
        sweep = 'sae_variants' 
    elif len(sae_variants)==1:
        sweep = 'edge_selection'
    else:
        raise ValueError("Please provide either SAE variants or edge selections, not both.")

    # Colors for the plots
    colors = ['ro-', 'go-', 'bo-', 'mo-', 'ko-', 'co-']

    
    kl_values_list = []
    feature_counts_list = []
    if sweep == 'sae_variants':
        method_labels = sae_variants
        edge_selection = edge_selections[0]

        for method in method_labels:
            # Load experiments and extract logits
            layer_logits_values, feature_counts = load_experiments_and_extract_data(
                exp_output,
                data_dir, 
                method,
                edge_selection,
                layers
            )
        
            # Store results
            kl_values_list.append(layer_logits_values)
            feature_counts_list.append(feature_counts)

    elif sweep == 'edge_selection':
        method_labels = edge_selections
        sae_variant = sae_variants[0]

        for method in method_labels:
            # Load experiments and extract logits
            layer_logits_values, feature_counts = load_experiments_and_extract_data(
                exp_output,
                data_dir, 
                sae_variant,
                method,
                layers
            )

            # Store results
            kl_values_list.append(layer_logits_values)
            feature_counts_list.append(feature_counts)
        
    else:
        raise ValueError("Invalid sweep type. Use 'sae_variants' or 'edge_selection'.")

    # PLOTTING

    # If token_idx is not None, average over all token positions
    token_idx = None

    # Lists to store results
    mean_values = []
    yerr_lower_values = []
    yerr_upper_values = []

    # Calculate statistics
    for layer_idx in range(len(layers)):
        layer_means = []
        layer_yerr_lower = []
        layer_yerr_upper = []
        
        for method_idx in range(len(method_labels)):
            method_means = []
            method_yerr_lower = []
            method_yerr_upper = []
            
            for edge_idx in range(len(feature_counts_list[method_idx])):
                # If token_idx is not None, average over all token positions
                if token_idx is None:
                    # Flatten the tensor to average over all dimensions
                    tensor = kl_values_list[method_idx][layer_idx][edge_idx]
                    flat_data = tensor.flatten()
                    if isinstance(flat_data, torch.Tensor):
                        flat_data = flat_data.numpy()
                else:
                    # Flatten the tensor to average over specific token positions
                    tensor = kl_values_list[method_idx][layer_idx][edge_idx][:, token_idx]
                    flat_data = tensor.flatten()
                    if isinstance(flat_data, torch.Tensor):
                        flat_data = flat_data.numpy()
                
                # Compute mean and confidence interval
                mean_value = np.mean(flat_data)
                lower_ci, upper_ci = bootstrap_ci(flat_data)
                yerr_lower = mean_value - lower_ci
                yerr_upper = upper_ci - mean_value
                
                # Store values
                method_means.append(mean_value)
                method_yerr_lower.append(yerr_lower)
                method_yerr_upper.append(yerr_upper)
            
            layer_means.append(method_means)
            layer_yerr_lower.append(method_yerr_lower)
            layer_yerr_upper.append(method_yerr_upper)
        
        mean_values.append(layer_means)
        yerr_lower_values.append(layer_yerr_lower)
        yerr_upper_values.append(layer_yerr_upper)

    # Plot results for each layer
    plt.figure(figsize=(10, 6))
    for layer_idx in range(len(layers)):
        plt.figure(figsize=(10, 6))
        for method_idx in range(len(method_labels)):
            plt.errorbar(
            feature_counts_list[method_idx], 
            mean_values[layer_idx][method_idx],
            yerr=[yerr_lower_values[layer_idx][method_idx], yerr_upper_values[layer_idx][method_idx]],
            fmt=colors[method_idx],
            capsize=3,
            label=method_labels[method_idx]
            )

        plt.xlabel('Number of Features')
        plt.ylabel('KL Divergence')
        plt.title(f'Layer {layers[layer_idx]}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path.cwd() / 'xavier' / 'experiments' / 'plots' / f'{run_idx}'
        output_path = output_dir / f'{sweep}_{layers[layer_idx]}_{timestamp}.png'
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    main()
