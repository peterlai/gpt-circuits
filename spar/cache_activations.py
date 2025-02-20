"""
Extracting and caching activations from a GPT model.

Example usage:
$ python -m spar.cache_activations --model=shakespeare_64x4 --dataset=shakespeare --batch_size=64
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from config.gpt.training import options
from models.gpt import GPT
import warnings
import argparse

def cache_activations(dataset_path, model_path, output_dir=None, batch_size=64, device="cuda"):
    """
    Process a dataset through a GPT model, cache residual activations, and save to disk.
    
    Args:
        dataset_path (str): Path to the dataset directory containing train_000000.npy and val_000000.npy
        model_path (str): Path to the model checkpoint directory
        output_dir (str, optional): Directory to save cached activations. Defaults to data/<model_name>
        batch_size (int): Batch size for processing the dataset
        device (str): Device to run the model on (e.g., "cuda", "cpu")
    
    Returns:
        dict: Dictionary with 'train' and 'val' keys containing cached activations
    """
    # Extract model name from model_path
    model_name = os.path.basename(model_path)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join("data", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model configuration
    config = options[model_name]
    model = GPT(config.gpt_config)
    model = model.load(model_path, device=config.device)
    model.to(device)

    # Disable gradient calculation
    torch.set_grad_enabled(False)

    # Process both train and val datasets
    cached_activations = {}
    for split in ["train", "val"]:
        data_path = os.path.join(dataset_path, f"{split}_000000.npy")
        if not os.path.exists(data_path):
            print(f"Skipping {split} split: {data_path} not found")
            continue

        # Load and prepare data
        np_data = np.load(data_path, allow_pickle=False).astype(np.int32)
        tensor_data = torch.tensor(np_data, dtype=torch.long)

        # Check if data length is not a multiple of block size
        if len(tensor_data) % model.config.block_size != 0:
            original_len = len(tensor_data)
            new_len = (len(tensor_data) // model.config.block_size) * model.config.block_size
            truncated_count = original_len - new_len
            warnings.warn(
                f"{split} data length {original_len} is not a multiple of block size {model.config.block_size}. "
                f"Truncating to {new_len} elements, losing {truncated_count} elements."
            )
            tensor_data = tensor_data[:new_len]

        data = tensor_data.reshape(-1, model.config.block_size).to(device)

        # Process in batches
        batches = torch.chunk(data, len(data) // batch_size)
        all_caches = []

        for batch in tqdm(batches, desc=f"Processing {split} split"):
            _, _, cache = model(batch, cache_resid=True)
            # Stack cache entries (assuming cache is a list of tensors from each layer)
            # Adjust this if your cache structure differs
            all_caches.append(cache.cpu())  # Move to CPU to save GPU memory

        # Concatenate all batches along the batch dimension
        cached_activations[split] = torch.cat(all_caches, dim=0)  # Shape: (total_batches*B, num_layers,T, n_embd)

        # Save to disk
        output_path = os.path.join(output_dir, f"{split}_activations.npy")
        np_cache = cached_activations[split].numpy()
        np.save(output_path, np_cache)
        print(f"Saved {split} activations to {output_path}")

    return cached_activations

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Cache GPT model activations")
    parser.add_argument("--model", required=True, help="Model name (e.g., shakespeare_64x4)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., shakespeare)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing (default: 64)")
    parser.add_argument("--output_dir", default=None, help="Output directory for cached activations (optional)")
    parser.add_argument("--cuda_device", default="0", help="CUDA device number (default: 0)")

    # Parse arguments
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Add root directory to sys.path dynamically
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Define paths
    checkpoints = "checkpoints"
    data_root = "data"
    model_path = os.path.join(checkpoints, args.model)
    dataset_path = os.path.join(data_root, args.dataset)

    # Run the function
    activations = cache_activations(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )