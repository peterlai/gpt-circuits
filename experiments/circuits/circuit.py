"""
Find and export the circuit needed to reconstruct the output logits of a model to within a certain KL divergence
threshold.

$ python -m experiments.circuits.circuit --split=val --sequence_idx=5120 --token_idx=15
"""

import argparse
from pathlib import Path

import torch

from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.circuits import CircuitSearch
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_idx", type=int, help="Index for start of sequence [0...shard.tokens.size)")
    parser.add_argument("--token_idx", type=int, help="Index for token in the sequence [0...block_size)")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    parser.add_argument("--threshold", type=float, default=0.1, help="Max threshold for KL divergence")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    threshold = args.threshold
    shard_token_idx = args.sequence_idx
    target_token_idx = args.token_idx


    # Set paths
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    dirname = f"{args.split}.{args.shard_idx}.{shard_token_idx}.{target_token_idx}"
    circuit_dir = checkpoint_dir / "circuits" / dirname

    # Load model
    defaults = Config()
    model: SparsifiedGPT = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore

    # Load cached metrics and feature magnitudes
    model_profile = ModelProfile(checkpoint_dir)
    model_cache = ModelCache(checkpoint_dir)

    # Set feature ablation strategy
    num_samples = 256  # Number of samples to use for estimating KL divergence
    k_nearest = 256  # How many nearest neighbors to consider in resampling
    positional_coefficient = 2.0  # How important is the position of a feature
    ablator = ResampleAblator(
        model_profile,
        model_cache,
        k_nearest=k_nearest,
        positional_coefficient=positional_coefficient,
    )

    # Load shard
    shard = DatasetShard(dir_path=Path(args.data_dir), split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    tokenizer = model.gpt.config.tokenizer
    tokens: list[int] = shard.tokens[args.sequence_idx : args.sequence_idx + model.config.block_size].tolist()
    decoded_tokens = tokenizer.decode_sequence(tokens)
    decoded_target = tokenizer.decode_token(tokens[target_token_idx])
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")
    print(f"Target threshold: {threshold}")

    # Start search
    circuit_search = CircuitSearch(model, ablator, num_samples)
    circuit = circuit_search.search(tokens, target_token_idx, threshold)

    # Print results, grouping nodes by layer
    print("\nCircuit:")
    for layer_idx in range(model.gpt.config.n_layer + 1):
        layer_nodes = [node for node in circuit.nodes if node.layer_idx == layer_idx]
        layer_nodes.sort(key=lambda node: node.token_idx)
        print(f"Layer {layer_idx}: {layer_nodes}")