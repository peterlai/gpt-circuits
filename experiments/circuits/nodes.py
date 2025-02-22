"""
Find and export all circuit nodes needed to reconstruct the output logits of a model to within a certain KL divergence
threshold.

$ python -m experiments.circuits.nodes --sequence_idx=0 --token_idx=51 --layer_idx=0
"""

import argparse
from collections import defaultdict

import torch

from circuits import json_prettyprint
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator, ZeroAblator  # noqa: F401
from circuits.search.nodes import NodeSearch
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
    parser.add_argument("--layer_idx", type=int, default=0, help="SAE layer to analyze")
    parser.add_argument("--threshold", type=float, default=0.2, help="Max threshold for KL divergence")
    parser.add_argument("--start_from", type=int, default=0, help="Index of token to start search from")
    parser.add_argument("--resample", action=argparse.BooleanOptionalAction, default=True, help="Use resampling")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    threshold = args.threshold
    layer_idx = args.layer_idx
    target_token_idx = args.token_idx
    start_token_idx = args.start_from

    # Set paths
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    dirname = f"{args.split}.{args.shard_idx}.{args.sequence_idx}.{args.token_idx}"
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
    if not args.resample:
        num_samples = 1  # Number of samples to use for estimating KL divergence
        ablator = ZeroAblator()
    else:
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
    shard = DatasetShard(dir_path=args.data_dir, split=args.split, shard_idx=args.shard_idx)

    # Get token sequence
    tokenizer = model.gpt.config.tokenizer
    tokens: list[int] = shard.tokens[args.sequence_idx : args.sequence_idx + model.config.block_size].tolist()
    decoded_tokens = tokenizer.decode_sequence(tokens)
    decoded_target = tokenizer.decode_token(tokens[target_token_idx])
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")
    print(f"Target layer: {layer_idx}")
    print(f"Target threshold: {threshold}")

    # Start search
    node_search = NodeSearch(model, ablator, num_samples)
    circuit_nodes = node_search.search(tokens, layer_idx, start_token_idx, target_token_idx, threshold)

    # Group features by token idx
    grouped_nodes = defaultdict(list)
    for node in sorted(circuit_nodes):
        grouped_nodes[node.token_idx].append(node.feature_idx)

    # Export circuit features
    data = {
        "data_dir": args.data_dir,
        "split": args.split,
        "shard_idx": args.shard_idx,
        "sequence_idx": args.sequence_idx,
        "token_idx": target_token_idx,
        "layer_idx": layer_idx,
        "threshold": threshold,
        "nodes": grouped_nodes,
    }
    circuit_dir.mkdir(parents=True, exist_ok=True)
    with open(circuit_dir / f"nodes.{layer_idx}.json", "w") as f:
        f.write(json_prettyprint(data))
