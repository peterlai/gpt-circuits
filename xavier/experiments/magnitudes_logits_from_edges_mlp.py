import os
import sys  # Add this import
import torch
import random
import time
from pathlib import Path
import numpy as np
from transformer_lens.hook_points import HookPoint
import torch.nn.functional as F

# Path setup
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.sae.topk import StaircaseTopKSAE
from models.gpt import GPT
from models.mlpsparsified import MLPSparsifiedGPT
from data.tokenizers import ASCIITokenizer
from david.convert_to_tl import convert_gpt_to_transformer_lens
from david.convert_to_tl import run_tests as run_tl_tests
from xavier.utils import create_tokenless_edges_from_array

@torch.no_grad()
def main():

    # Parameters
    num_edges = 1000
    upstream_layer_num = 0
    num_prompts = 3
    edge_selection = 'random'
    seed = 25


    # Set random seed
    random.seed(seed)

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Setup model paths
    checkpoint_dir = project_root / "checkpoints"
    gpt_dir = checkpoint_dir / "shakespeare_64x4"
    mlp_dir = checkpoint_dir / "mlp-topk.shakespeare_64x4"
    data_dir = project_root / "data"

    # Load GPT model
    print("Loading GPT model...")
    gpt = GPT.load(gpt_dir, device=device)

    sae_config =SAEConfig(
            gpt_config=gpt.config,
            n_features=tuple(64 * n for n in (8,8,8,8,8,8,8,8)),
            sae_variant=SAEVariant.TOPK,
            top_k = (10, 10, 10, 10, 10, 10, 10, 10)
        )

    # Load GPT MLP
    print("Loading sparsified MLP model...")
    loss_coefficients = LossCoefficients()
    model = MLPSparsifiedGPT.load(mlp_dir, 
                                    loss_coefficients,
                                    trainable_layers = None,
                                    device = device)
    model.to(device)

    # Load validation data
    val_data_dir = data_dir / 'shakespeare/val_000000.npy'
    with open(val_data_dir, 'rb') as f:
        val_array = np.load(f)
    val_tensor = torch.from_numpy(val_array).to(torch.long)  # Convert to long tensor
 
    # Calculate number of complete chunks we can make
    sequence_length = 128 # Hardcoded for now
    target_token_idx = sequence_length - 1

    num_chunks = val_tensor.shape[0] // sequence_length
    usable_data = val_tensor[:num_chunks * sequence_length]
    reshaped_val_tensor = usable_data.reshape(num_chunks, sequence_length)
    input_ids = reshaped_val_tensor[:num_prompts, :].to(device)  # Also move to the correct device

    print(f"Computing upstream & downstream magnitudes (full circuit)...")
    keys = [f'{upstream_layer_num}_mlpin', f'{upstream_layer_num}_mlpout']
    with model.use_saes(activations_to_patch = keys) as encoder_outputs:
        _ = model(input_ids)
        upstream_magnitudes = encoder_outputs[keys[0]].feature_magnitudes
        downstream_magnitudes_full_circuit = encoder_outputs[keys[1]].feature_magnitudes

    # Create edges
    num_upstream_features = model.config.n_features[upstream_layer_num]
    num_downstream_features = model.config.n_features[upstream_layer_num + 1]
    
    print(f"Creating {num_edges} edges using {edge_selection} selection strategy...")
    
    # Create edge array based on selection strategy
    if edge_selection == "random":
        # Create a random permutation of all possible edges
        all_edges = [(a, b) for a in range(num_upstream_features) for b in range(num_downstream_features)]
        random.shuffle(all_edges)
        edge_arr = all_edges[:num_edges]

    # Create TokenlessEdge objects
    edges = create_tokenless_edges_from_array(edge_arr, upstream_layer_num)
    
    # Compute downstream magnitudes from edges
    print(f"Computing downstream magnitudes from {len(edges)} edges...")
    start_time = time.time()
    
    if num_edges == num_upstream_features * num_downstream_features:
        # Use the full circuit magnitudes
        print(f"Using full circuit magnitudes...")
        downstream_magnitudes = downstream_magnitudes_full_circuit
    else:
        # Compute downstream magnitudes from edges
        # WRITE THIS FUNCTION
        downstream_magnitudes = torch.zeros_like(downstream_magnitudes_full_circuit)

    # Prepare data to compute logits
    with model.record_activations() as activations:
            with model.use_saes() as encoder_outputs:
                _, _ = model.gpt(input_ids, targets=None)
    
    layer_idx, hook_loc = model.split_sae_key(f'{upstream_layer_num}_mlpout')
    resid_mid = activations[f'{layer_idx}_residmid']

    assert upstream_layer_num == layer_idx, f"Upstream layer number {upstream_layer_num} does not match layer index {layer_idx}"

    # Compute logits subcircuit
    x_reconstructed = model.saes[f'{upstream_layer_num}_mlpout'].decode(downstream_magnitudes) 
    predicted_logits = model.gpt.forward_with_patched_activations(
        x_reconstructed, resid_mid, layer_idx, hook_loc
    )   # Shape: (num_batches, T, V)
    print(predicted_logits.shape)

    # Compute logits full circuit
    x_reconstructed_full_circuit = model.saes[f'{upstream_layer_num}_mlpout'].decode(downstream_magnitudes_full_circuit) 
    predicted_logits_full_circuit = model.gpt.forward_with_patched_activations(
        x_reconstructed_full_circuit, resid_mid, layer_idx, hook_loc
    )  # Shape: (num_batches, T, V)
    print(predicted_logits_full_circuit.shape)

    # Compute KL divergence between full and subcircuit logits
    probs_full_circuit = F.softmax(predicted_logits_full_circuit, dim=-1)
    probs = F.softmax(predicted_logits, dim=-1)

    # Compute KL divergence: KL(P||Q) = sum_i P(i) * log(P(i)/Q(i))
    epsilon = 1e-8
    kl_div = probs_full_circuit * torch.log((probs_full_circuit + epsilon) / (probs + epsilon))
    kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary dimension

    print(f"KL divergence shape: {kl_div.shape}")

    # Compute the time taken for the computation
    execution_time = time.time() - start_time
    print(f"Computation completed in {execution_time:.2f} seconds")


    print("Upstream Magnitudes:")
    print(upstream_magnitudes.shape)

    print("Downstream Magnitudes:")
    print(downstream_magnitudes.shape)

if __name__ == "__main__":
    print('starting...')
    main()
    print('done...')