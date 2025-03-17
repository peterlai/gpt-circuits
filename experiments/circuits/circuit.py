"""
Find and export the circuit needed to reconstruct the output logits of a model to within a certain KL divergence
threshold.

$ python -m experiments.circuits.circuit --text="Are you now going to discredit him?" --token_idx=28
$ python -m experiments.circuits.circuit --split=val --sequence_idx=5120 --token_idx=15
"""

import argparse
import dataclasses
import hashlib
from collections import defaultdict
from pathlib import Path

import torch

from circuits import SearchConfiguration, json_prettyprint
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.circuits import CircuitSearch
from circuits.search.divergence import get_predictions
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    # For analyzing a custom sequence
    parser.add_argument("--text", type=str, help="Custom text to analyze")
    # For analyzing a specific sequence in a dataset
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--sequence_idx", type=int, help="Index for start of sequence [0...shard.tokens.size)")
    parser.add_argument("--token_idx", type=int, help="Index for token in the sequence [0...block_size)")
    parser.add_argument("--skip_edges", action="store_true", help="Produce placeholder values for edges")
    parser.add_argument("--config_name", type=str, help="Search configuration name")
    return parser.parse_args()


def load_tokens(model: SparsifiedGPT, args: argparse.Namespace) -> tuple[list[int], str]:
    """
    Return the tokens to use for the circuit search.

    :param args: Command line arguments
    :return: Tuple of tokens and recommended dirname
    """
    if text := args.text:
        # Tokenize custom text
        trimmed_text = (text + " " * model.config.block_size)[: model.config.block_size]
        tokenizer = model.gpt.config.tokenizer
        tokens = tokenizer.encode(trimmed_text)[: model.config.block_size]
        hash_prefix = hashlib.sha256(text.encode("utf-8")).hexdigest()[:7]
        dirname = f"{hash_prefix}.{args.token_idx}"
        return tokens, dirname
    else:
        # Load tokens from dataset shard
        shard_token_idx = args.sequence_idx
        shard = DatasetShard(dir_path=Path(args.data_dir), split=args.split, shard_idx=args.shard_idx)
        tokens: list[int] = shard.tokens[shard_token_idx : shard_token_idx + model.config.block_size].tolist()
        dirname = f"{args.split}.{args.shard_idx}.{shard_token_idx}.{args.token_idx}"
        return tokens, dirname


def load_configuration(config_name: str) -> SearchConfiguration:
    """
    Load the search configuration from a configuration name.
    """
    match config_name or "":
        case x if x.endswith("-cluster"):
            return SearchConfiguration(
                threshold=0.25,
            )
        case x if x.endswith("-cluster-nopos"):
            return SearchConfiguration(
                threshold=0.25,
                max_positional_coefficient=0.0,  # Disable positional coefficient
            )
        case x if x.endswith("-random"):
            return SearchConfiguration(
                threshold=0.25,
                k_nearest=None,
                max_positional_coefficient=0.0,  # Disable positional coefficient
            )
        case x if x.endswith("-random-pos"):
            return SearchConfiguration(
                threshold=0.25,
                k_nearest=None,
            )
        case x if x.endswith("-zero"):
            return SearchConfiguration(
                threshold=0.25,
                k_nearest=0,
                num_edge_samples=1,  # Resampling isn't needed
                num_node_samples=1,  # Resampling isn't needed
            )
        case _:
            return SearchConfiguration()


def main():
    # Parse command line arguments
    args = parse_args()
    target_token_idx = args.token_idx
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model

    # Load model
    defaults = Config()
    model: SparsifiedGPT = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore
        torch.set_float32_matmul_precision("high")

    # Load tokens
    tokens, dirname = load_tokens(model, args)

    # Set output directory
    circuit_dir = checkpoint_dir / "circuits" / dirname

    # Setup search configuration
    config = load_configuration(args.config_name)

    # Load cached metrics and feature magnitudes
    model_profile = ModelProfile(checkpoint_dir)
    model_cache = ModelCache(checkpoint_dir)

    # Get token sequence
    tokenizer = model.gpt.config.tokenizer
    decoded_tokens = tokenizer.decode_sequence(tokens)
    printable_tokens = decoded_tokens.replace("\n", "\\n")
    decoded_target = tokenizer.decode_token(tokens[target_token_idx])
    print(f'Using sequence: "{printable_tokens}"')
    print(f"Target token: `{decoded_target}` at index {args.token_idx}")
    print(f"Target threshold: {config.threshold}")

    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get target logits
    with torch.no_grad():
        target_logits = model(input).logits.squeeze(0)[target_token_idx]  # Shape: (V)
        target_predictions = get_predictions(tokenizer, target_logits)
        print(f"Target predictions: {target_predictions}")

    # Start search
    circuit_search = CircuitSearch(
        model,
        model_profile,
        model_cache,
        config=config,
    )
    search_result = circuit_search.search(tokens, target_token_idx, skip_edges=args.skip_edges)

    # Export circuit search configuration
    config_path = circuit_dir / "config.json"
    circuit_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        data = {
            "tokens": tokens,
            "target_token_idx": target_token_idx,
            "num_nodes": len(search_result.circuit.nodes),
            "predictions": target_predictions,
            "klds": search_result.klds,
            "search_config": dataclasses.asdict(config),
        }
        f.write(json_prettyprint(data))

    # Print circuit nodes, grouping nodes by layer
    print("\nCircuit nodes:")
    for layer_idx in range(model.gpt.config.n_layer + 1):
        layer_nodes = [node for node in search_result.circuit.nodes if node.layer_idx == layer_idx]
        layer_nodes.sort(key=lambda node: node.token_idx)
        print(f"Layer {layer_idx}: {layer_nodes}")

        # Group features by token idx
        grouped_nodes = defaultdict(dict)
        for node in layer_nodes:
            # Map feature indices to rank
            grouped_nodes[node.token_idx][node.feature_idx] = search_result.node_ranks[node]

        # Positional coefficient used for clustering
        positional_coefficient = search_result.positional_coefficients[layer_idx]

        # Export layer
        data = {
            "layer_idx": layer_idx,
            "positional_coefficient": positional_coefficient,
            "predictions": search_result.predictions[layer_idx],
            "nodes": grouped_nodes,
        }
        circuit_dir.mkdir(parents=True, exist_ok=True)
        with open(circuit_dir / f"nodes.{layer_idx}.json", "w") as f:
            f.write(json_prettyprint(data))

    # Save edge data
    for layer_idx in range(1, model.gpt.config.n_layer + 1):
        downstream_nodes = {n for n in search_result.circuit.nodes if n.layer_idx == layer_idx}

        # Group edges by downstream token
        grouped_edges = {}
        for downstream_node in sorted(downstream_nodes):
            edges = [
                (edge, value)
                for edge, value in search_result.edge_importances.items()
                if edge.downstream == downstream_node
            ]
            upstream_to_value = {}
            for edge, value in sorted(edges, key=lambda x: x[0]):
                upstream_to_value[".".join(map(str, edge.upstream.as_tuple()))] = round(value, 5)
            grouped_edges[".".join(map(str, downstream_node.as_tuple()))] = upstream_to_value

        # Upstream token importance
        upstream_tokens = defaultdict(dict)
        upstream_edge_groups = {
            edge_group
            for edge_group in search_result.token_importances.keys()
            if edge_group.downstream_layer_idx == layer_idx
        }
        for edge_group in upstream_edge_groups:
            downstream_key = f"{edge_group.downstream_layer_idx}.{edge_group.downstream_token_idx}"
            upstream_key = f"{edge_group.upstream_layer_idx}.{edge_group.upstream_token_idx}"
            token_importance = search_result.token_importances[edge_group]
            upstream_tokens[downstream_key][upstream_key] = round(token_importance, 5)

        # Export circuit features
        data = {
            "layer_idx": layer_idx,
            "edges": grouped_edges,
            "tokens": upstream_tokens,
        }
        circuit_dir.mkdir(parents=True, exist_ok=True)
        with open(circuit_dir / f"edges.{layer_idx}.json", "w") as f:
            f.write(json_prettyprint(data))

    print(f"\nExported circuit to {circuit_dir}")


if __name__ == "__main__":
    main()
