"""
Find and export all circuit edges between two layers needed to reconstruct the output logits of a model to within a
certain KL divergence threshold.

$ python -m experiments.circuits.edges --circuit=train.0.0.51 --upstream_layer=0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from circuits import Edge, Node, json_prettyprint
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator, ZeroAblator  # noqa: F401
from circuits.search.edges import EdgeSearch
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    parser.add_argument("--circuit", type=str, default="train.0.0.51", help="Circuit directory name")
    parser.add_argument("--upstream_layer", type=int, default=0, help="Find edges from this layer")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for MSE increase (e.g.: 0.1 = 10%)")
    parser.add_argument("--resample", action=argparse.BooleanOptionalAction, default=True, help="Use resampling")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    threshold = args.threshold

    # Set paths
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    circuit_dir = Path("exports") / args.model / "circuits" / args.circuit
    upstream_path = circuit_dir / f"nodes.{args.upstream_layer}.json"
    downstream_path = circuit_dir / f"nodes.{args.upstream_layer + 1}.json"

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
        k_nearest = 256  # How many nearest neighbors to consider in resampling
        num_samples = 256  # Number of samples to use for estimating KL divergence
        positional_coefficient = 2.0  # How important is the position of a feature
        ablator = ResampleAblator(
            model_profile,
            model_cache,
            k_nearest=k_nearest,
            positional_coefficient=positional_coefficient,
        )

    # Load sequence args
    with open(upstream_path) as f:
        upstream_json = json.load(f)
        data_dir = upstream_json["data_dir"]
        split = upstream_json["split"]
        shard_idx = upstream_json["shard_idx"]
        sequence_idx = upstream_json["sequence_idx"]
        target_token_idx = upstream_json["token_idx"]
        upstream_layer_idx = upstream_json["layer_idx"]

    # Load shard
    shard = DatasetShard(dir_path=data_dir, split=split, shard_idx=shard_idx)

    # Get token sequence
    tokenizer = model.gpt.config.tokenizer
    tokens: list[int] = shard.tokens[sequence_idx : sequence_idx + model.config.block_size].tolist()
    decoded_tokens = tokenizer.decode_sequence(tokens)
    decoded_target = tokenizer.decode_token(tokens[target_token_idx])
    print(f'Using sequence: "{decoded_tokens.replace("\n", "\\n")}"')
    print(f"Target token: `{decoded_target}` at index {target_token_idx}")
    print(f"Target layer: {upstream_layer_idx}")
    print(f"Target threshold: {threshold}")

    # Load upstream nodes
    upstream_nodes = set()
    with open(upstream_path) as f:
        data = json.load(f)
        for token_idx, feature_idxs in data["nodes"].items():
            for feature_idx in feature_idxs:
                upstream_nodes.add(Node(upstream_layer_idx, int(token_idx), feature_idx))
    upstream_nodes = frozenset(upstream_nodes)

    # Load downstream nodes
    downstream_nodes = set()
    with open(downstream_path) as f:
        data = json.load(f)
        for token_idx, feature_idxs in data["nodes"].items():
            for feature_idx in feature_idxs:
                downstream_nodes.add(Node(upstream_layer_idx + 1, int(token_idx), feature_idx))
    downstream_nodes = frozenset(downstream_nodes)

    # Start search
    edge_search = EdgeSearch(model, model_profile, ablator, num_samples)
    circuit_edges: frozenset[Edge] = edge_search.search(
        tokens, target_token_idx, upstream_nodes, downstream_nodes, threshold
    )

    # Group edges by downstream token
    grouped_edges = defaultdict(list)
    for downstream_node in sorted(downstream_nodes):
        edges = [edge for edge in circuit_edges if edge.downstream == downstream_node]
        grouped_edges[".".join(map(str, downstream_node.as_tuple()))] = [
            ".".join(map(str, edge.upstream.as_tuple())) for edge in sorted(edges)
        ]

    # Export circuit features
    data = {
        "data_dir": data_dir,
        "split": split,
        "shard_idx": shard_idx,
        "sequence_idx": sequence_idx,
        "token_idx": target_token_idx,
        "layer_idx": upstream_layer_idx + 1,
        "threshold": threshold,
        "edges": grouped_edges,
    }
    circuit_dir.mkdir(parents=True, exist_ok=True)
    with open(circuit_dir / f"edges.{upstream_layer_idx + 1}.json", "w") as f:
        f.write(json_prettyprint(data))
