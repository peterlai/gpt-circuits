"""
Export circuit for visualization using Node app.

$ python -m experiments.circuits.export --circuit=train.0.0.51 --dirname=toy-resample
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from circuits import Circuit, Edge, Node, json_prettyprint
from circuits.features.cache import ModelCache
from circuits.features.profiles import FeatureProfile, ModelProfile
from circuits.features.samples import ModelSampleSet, Sample
from circuits.search.ablation import ResampleAblator
from circuits.search.clustering import ClusterSearch
from circuits.search.divergence import (
    get_predicted_logits,
    get_predictions,
    patch_feature_magnitudes,
)
from config import Config, TrainingConfig
from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model name")
    parser.add_argument("--circuit", type=str, default="train.0.0.51", help="Circuit directory name")
    parser.add_argument("--dirname", type=str, help="Output directory name")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set paths
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    circuit_dir = checkpoint_dir / "circuits" / args.circuit
    base_dir = Path("app/public/samples") / args.dirname

    # Load model
    defaults = Config()
    model: SparsifiedGPT = SparsifiedGPT.load(checkpoint_dir, device=defaults.device).to(defaults.device)
    model.eval()

    # Compile if enabled
    if defaults.compile:
        model = torch.compile(model)  # type: ignore

    # Load cached metrics and feature samples
    model_profile = ModelProfile(checkpoint_dir)
    model_cache = ModelCache(checkpoint_dir)
    model_sample_set = ModelSampleSet(checkpoint_dir=TrainingConfig.checkpoints_dir / args.model)

    # Load sequence args
    with open(circuit_dir / "nodes.0.json") as f:
        data = json.load(f)
        data_dir = data["data_dir"]
        split = data["split"]
        shard_idx = data["shard_idx"]
        sequence_idx = data["sequence_idx"]
        target_token_idx = data["token_idx"]

    # Load shard
    shard = DatasetShard(data_dir, split, shard_idx)

    # Get tokens
    tokens: list[int] = shard.tokens[sequence_idx : sequence_idx + model.config.block_size].tolist()

    # Gather circuit nodes
    nodes: set[Node] = set()
    for layer_idx in range(model.gpt.config.n_layer + 1):
        with open(circuit_dir / f"nodes.{layer_idx}.json", "r") as f:
            data = json.load(f)
            for token_str, feature_idxs in data["nodes"].items():
                token_idx = int(token_str)
                for feature_idx in feature_idxs:
                    node = Node(layer_idx, token_idx, feature_idx)
                    nodes.add(node)

    # Gather circuit edges
    edges: set[Edge] = set()
    for layer in range(1, model.gpt.config.n_layer + 1):
        with open(circuit_dir / f"edges.{layer}.json", "r") as f:
            data = json.load(f)
            for edge_key, upstream_node_keys in data["edges"].items():
                downstream_node = Node(*map(int, edge_key.split(".")))
                for upstream_node_key in upstream_node_keys:
                    upstream_node = Node(*map(int, upstream_node_key.split(".")))
                    edges.add(Edge(upstream_node, downstream_node))

    # Export features
    export_features(
        base_dir / "features",
        nodes,
        model,
        model_profile,
        model_sample_set,
        shard,
    )

    # Export data.json
    export_circuit_data(
        base_dir / "samples" / str(sequence_idx + target_token_idx),
        model,
        model_profile,
        model_cache,
        nodes,
        edges,
        shard,
        sequence_idx,
        target_token_idx,
    )

    # Export similar.json
    export_similar_samples(
        base_dir / "samples" / str(sequence_idx + target_token_idx),
        model,
        model_profile,
        model_cache,
        nodes,
        shard,
        tokens,
        target_token_idx,
    )


def export_similar_samples(
    samples_dir: Path,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    nodes: set[Node],
    shard: DatasetShard,
    tokens: list[int],
    target_token_idx: int,
):
    """
    Export similar samples to similar.json
    """
    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get target feature magnitudes
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(input)
    last_layer_idx = model.gpt.config.n_layer
    target_feature_magnitudes = output.feature_magnitudes[last_layer_idx][0, target_token_idx, :].cpu().numpy()
    target_nodes = [n for n in nodes if n.token_idx == target_token_idx and n.layer_idx == last_layer_idx]
    circuit_feature_idxs = np.array([node.feature_idx for node in nodes if node in target_nodes])

    # Get samples that are similar to the target token
    cluster_search = ClusterSearch(model_profile, model_cache)
    cluster = cluster_search.get_cluster(
        last_layer_idx,
        target_token_idx,
        target_feature_magnitudes,
        circuit_feature_idxs,
        k_nearest=25,
        positional_coefficient=0.0,
    )
    cluster_samples = cluster.as_sample_set().samples

    # Data to export
    data = {
        "samples": [],
        "decodedTokens": [],
        "tokenIdxs": [],
        "absoluteTokenIdxs": [],
        "magnitudeIdxs": [],
        "magnitudeValues": [],
        "maxActivation": 1.0,
    }

    # Add samples
    tokenizer = model.gpt.config.tokenizer
    for sample in cluster_samples:
        starting_idx = sample.block_idx * model.config.block_size
        tokens = shard.tokens[starting_idx : starting_idx + model.config.block_size].tolist()
        decoded_sample = tokenizer.decode_sequence(tokens)
        decoded_tokens = [tokenizer.decode_token(token) for token in tokens]
        data["samples"].append(decoded_sample)
        data["decodedTokens"].append(decoded_tokens)
        data["tokenIdxs"].append(sample.token_idx)
        data["absoluteTokenIdxs"].append(starting_idx + sample.token_idx)
        data["magnitudeIdxs"].append(sample.magnitudes.nonzero()[1].tolist())
        data["magnitudeValues"].append([round(magnitude, 3) for magnitude in sample.magnitudes.data.tolist()])

    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(samples_dir / "similar.json", "w") as f:
        f.write(json_prettyprint(data))


def export_circuit_data(
    sample_dir: Path,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    nodes: set[Node],
    edges: set[Edge],
    shard: DatasetShard,
    sequence_idx: int,
    target_token_idx: int,
):
    """
    Export sample data to data.json
    """

    tokenizer = model.gpt.config.tokenizer
    tokens: list[int] = shard.tokens[sequence_idx : sequence_idx + model.config.block_size].tolist()

    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get feature magnitudes
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(input)

    # Data to export
    data = {
        "text": tokenizer.decode_sequence(tokens),
        "decoded_tokens": [tokenizer.decode_token(token) for token in tokens],
        "target_idx": target_token_idx,
        "absolute_target_idx": sequence_idx + target_token_idx,
    }

    # Set feature magnitudes
    data["activations"] = {}
    data["normalizedActivations"] = {}
    for node in nodes:
        magnitude = output.feature_magnitudes[node.layer_idx][0, node.token_idx, node.feature_idx].item()
        norm_coefficient = 1.0 / model_profile[node.layer_idx][node.feature_idx].max
        data["activations"][node_to_key(node, target_token_idx)] = round(magnitude, 3)
        data["normalizedActivations"][node_to_key(node, target_token_idx)] = round(magnitude * norm_coefficient, 3)

    # Set probabilities
    logits = output.logits[0, target_token_idx, :]
    probabilities = get_predictions(model.gpt.config.tokenizer, logits, 128)
    data["probabilities"] = {k: round(v / 100.0, 3) for k, v in probabilities.items() if v > 0.1}

    # Set circuit probabilities
    ablator = ResampleAblator(model_profile, model_cache, 128, 0.0)
    circuit = Circuit(frozenset(nodes), frozenset(edges))
    last_layer_idx = model.gpt.config.n_layer
    feature_magnitudes = output.feature_magnitudes[last_layer_idx][0]
    patched_feature_magnitudes = patch_feature_magnitudes(
        ablator,
        last_layer_idx,
        target_token_idx,
        [circuit],
        feature_magnitudes,
        num_samples=128,
    )
    predicted_logits = get_predicted_logits(
        model,
        last_layer_idx,
        patched_feature_magnitudes,
        target_token_idx,
    )[circuit]
    predicted_probabilities = get_predictions(model.gpt.config.tokenizer, predicted_logits, 128)
    data["circuit_probabilities"] = {k: round(v / 100.0, 3) for k, v in predicted_probabilities.items() if v > 0.1}

    # Set root features
    data["root_features"] = {}
    num_layers = model.gpt.config.n_layer
    for feature_idx, magnitude in enumerate(output.feature_magnitudes[num_layers][0, target_token_idx, :].tolist()):
        if magnitude > 0:
            data["root_features"][feature_idx] = round(magnitude, 3)

    # Set ablation graph
    data["ablation_graph"] = {}
    for downstream_node in sorted(set(edge.downstream for edge in edges)):
        dependencies = []
        upstream_edges = [edge for edge in edges if edge.downstream == downstream_node]
        for edge in sorted(upstream_edges):
            edge_weight = 1.0  # TODO: Calculate edge weight
            dependencies.append([node_to_key(edge.upstream, target_token_idx), edge_weight])
        data["ablation_graph"][node_to_key(downstream_node, target_token_idx)] = dependencies

    # Set group alation graph
    data["group_ablation_graph"] = {}
    for downstream_node in sorted(set(edge.downstream for edge in edges)):
        groups = []
        downstream_feature_magnitude = output.feature_magnitudes[downstream_node.layer_idx][
            0, downstream_node.token_idx, downstream_node.feature_idx
        ].item()
        upstream_edges = [edge for edge in edges if edge.downstream == downstream_node]
        upstream_blocks = set((edge.upstream.layer_idx, edge.upstream.token_idx) for edge in upstream_edges)
        for layer_idx, token_idx in sorted(upstream_blocks):
            block_weight = downstream_feature_magnitude  # TODO: Calculate block weight
            groups.append([f"{target_token_idx - token_idx}.{layer_idx}", block_weight])
        data["group_ablation_graph"][node_to_key(downstream_node, target_token_idx)] = groups

    # Export to data.json
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "data.json", "w") as f:
        f.write(json_prettyprint(data))


def export_features(
    features_dir,
    nodes: set[Node],
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_sample_set: ModelSampleSet,
    shard: DatasetShard,
):
    """
    Create a JSON file with feature metrics for every feature in the circuit.
    """
    # Find unique features
    features: set[tuple[int, int]] = set()
    for node in nodes:
        features.add((node.layer_idx, node.feature_idx))
    sorted_features = sorted(list(features))

    for layer_idx, feature_idx in sorted_features:
        # Load feature metrics
        feature_profile: FeatureProfile = model_profile[layer_idx][feature_idx]

        # Data to export
        data = {
            "samples": [],
            "decodedTokens": [],
            "tokenIdxs": [],
            "absoluteTokenIdxs": [],
            "magnitudeIdxs": [],
            "magnitudeValues": [],
            "maxActivation": feature_profile.max,
            "activationHistogram": {
                "counts": feature_profile.histogram_counts,
                "binEdges": feature_profile.histogram_edges,
            },
        }

        # Load feature samples
        samples: list[Sample] = model_sample_set[layer_idx][feature_idx].samples
        block_size = int(samples[0].magnitudes.shape[-1])  # type: ignore

        # Load sample tokens
        sample_tokens: list[list[int]] = []
        for sample in samples:
            starting_idx = sample.block_idx * block_size
            tokens = shard.tokens[starting_idx : starting_idx + block_size].tolist()
            sample_tokens.append(tokens)

        # Add decoded samples
        tokenizer = model.gpt.config.tokenizer
        for tokens in sample_tokens:
            decoded_sample = tokenizer.decode_sequence(tokens)
            decoded_tokens = [tokenizer.decode_token(token) for token in tokens]
            data["samples"].append(decoded_sample)
            data["decodedTokens"].append(decoded_tokens)

        # Add token idxs
        for sample in samples:
            data["tokenIdxs"].append(sample.token_idx)
            data["absoluteTokenIdxs"].append(block_size * sample.block_idx + sample.token_idx)
            pass

        # Add token magnitudes
        for sample in samples:
            data["magnitudeIdxs"].append(sample.magnitudes.nonzero()[1].tolist())
            data["magnitudeValues"].append([round(magnitude, 3) for magnitude in sample.magnitudes.data.tolist()])

        # Create file for feature
        features_dir.mkdir(parents=True, exist_ok=True)
        with open(features_dir / f"{layer_idx}.{feature_idx}.json", "w") as f:
            f.write(json_prettyprint(data))


def node_to_key(node: Node, target_token_idx: int) -> str:
    """
    Convert a node to a string key.
    """
    token_offset = target_token_idx - node.token_idx
    return f"{token_offset}.{node.layer_idx}.{node.feature_idx}"


if __name__ == "__main__":
    main()
