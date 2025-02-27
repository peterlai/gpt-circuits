"""
Export circuit for visualization using Node app.

$ python -m experiments.circuits.export --circuit=train.0.0.51 --dirname=toy-local
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
from config.gpt.models import GPTConfig
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
    parser.add_argument("--name", type=str, default="", help="Sample name")
    parser.add_argument("--version", type=str, default="", help="Sample version")
    parser.add_argument("--threshold", type=float, default=0.2, help="Circuit KL divergence")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    sample_name = args.name if args.name else args.circuit
    sample_version = args.version if args.version else str(args.threshold)

    # Set paths
    checkpoint_dir = TrainingConfig.checkpoints_dir / args.model
    circuit_dir = checkpoint_dir / "circuits" / args.circuit
    base_dir = Path("app/public/samples") / args.dirname
    features_dir = base_dir / "features"
    sample_dir = base_dir / "samples" / sample_name / sample_version

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
    model_sample_set = ModelSampleSet(checkpoint_dir)

    # Load sequence args
    with open(circuit_dir / "nodes.0.json") as f:
        data = json.load(f)
        data_dir: Path = Path(data["data_dir"])
        split: str = data["split"]
        shard_idx: int = data["shard_idx"]
        sequence_idx: int = data["sequence_idx"]
        target_token_idx: int = data["token_idx"]

    # Get tokens
    shard_for_tokens = DatasetShard(data_dir, split, shard_idx)
    tokens: list[int] = shard_for_tokens.tokens[sequence_idx : sequence_idx + model.config.block_size].tolist()

    # Gather circuit nodes
    node_importance: dict[Node, float] = {}
    for layer_idx in range(model.gpt.config.n_layer + 1):
        with open(circuit_dir / f"nodes.{layer_idx}.json", "r") as f:
            data = json.load(f)
            for token_str, features in data["nodes"].items():
                token_idx = int(token_str)
                for feature_str, kld in features.items():
                    feature_idx = int(feature_str)
                    node = Node(layer_idx, token_idx, feature_idx)
                    node_importance[node] = kld

    # Gather circuit edges
    edge_importance: dict[Edge, float] = {}
    token_importance: dict[Node, dict[int, float]] = {}
    for layer in range(1, model.gpt.config.n_layer + 1):
        with open(circuit_dir / f"edges.{layer}.json", "r") as f:
            data = json.load(f)
            # Get edge importance
            for edge_key, upstream_nodes in data["edges"].items():
                downstream_node = Node(*map(int, edge_key.split(".")))
                for upstream_node_key, importance in upstream_nodes.items():
                    upstream_node = Node(*map(int, upstream_node_key.split(".")))
                    edge = Edge(upstream_node, downstream_node)
                    edge_importance[edge] = importance
            # Get token importance
            for edge_key, upstream_tokens in data["upstream_tokens"].items():
                downstream_node = Node(*map(int, edge_key.split(".")))
                for token_str, importance in upstream_tokens.items():
                    token_idx = int(token_str)
                    if downstream_node not in token_importance:
                        token_importance[downstream_node] = {}
                    token_importance[downstream_node][token_idx] = importance

    # Construct circuit
    circuit = construct_circuit(model.gpt.config, node_importance, edge_importance, args.threshold)

    # Export blocks
    export_blocks(
        sample_dir,
        model,
        model_profile,
        model_cache,
        circuit.nodes,
        data_dir,
        tokens,
        target_token_idx,
    )
    # Export features
    export_features(
        sample_dir,
        features_dir,
        circuit.nodes,
        model,
        model_profile,
        model_cache,
        model_sample_set,
        data_dir,
        tokens,
        target_token_idx,
    )

    # Export data.json
    export_circuit_data(
        tokens,
        sample_dir,
        model,
        model_profile,
        model_cache,
        circuit,
        edge_importance,
        token_importance,
        target_token_idx,
        args.threshold,
    )


def construct_circuit(gpt_config: GPTConfig, node_importance, edge_importance, threshold) -> Circuit:
    """
    Construct a circuit from nodes and edges given a KLD threshold.
    """
    nodes = set()
    edges = set()

    # Build circuit starting from the last layer
    for layer_idx in range(gpt_config.n_layer, -1, -1):
        layer_nodes = set([node for node in node_importance if node.layer_idx == layer_idx])
        layer_node_importance = {node: node_importance[node] for node in layer_nodes if node in layer_nodes}

        # Find largest KLD below threshold
        klds_below_threshold = [kld for kld in layer_node_importance.values() if kld < threshold]
        if klds_below_threshold:
            layer_threshold = max(kld for kld in klds_below_threshold)
        else:
            #  Use the smallest KLD if all are above threshold
            layer_threshold = min(layer_node_importance.values())

        # Filter nodes based various criteria
        for node, kld in layer_node_importance.items():
            # Include node if needed to satisfy threshold
            if kld >= layer_threshold:
                nodes.add(node)
            # Include node if a downstream node exists at the same token index
            elif node.token_idx in {n.token_idx for n in nodes if n.layer_idx == layer_idx + 1}:
                nodes.add(node)

    # Filter edges based on nodes and importance
    for edge, importance in edge_importance.items():
        if edge.upstream in nodes and edge.downstream in nodes and importance > 0.1:
            edges.add(edge)

    # Try to add an edge to nodes with no upstream connections
    for node in nodes:
        # Skip embedding nodes
        if node.layer_idx == 0:
            continue
        if not any(edge.downstream == node for edge in edges):
            # Find the most important edge
            candidates = [e for e in edge_importance.keys() if e.downstream == node and e.upstream in nodes]
            if best_candidate := max(candidates, key=lambda e: edge_importance[e], default=None):
                edges.add(best_candidate)

    # Try to add an edge to nodes with no downstream connections
    for node in nodes:
        # Skip root nodes
        if node.layer_idx == gpt_config.n_layer:
            continue
        if not any(edge.upstream == node for edge in edges):
            # Find the most important edge
            candidates = [e for e in edge_importance.keys() if e.upstream == node and e.downstream in nodes]
            if best_candidate := max(candidates, key=lambda e: edge_importance[e], default=None):
                edges.add(best_candidate)

    # Create circuit
    circuit = Circuit(frozenset(nodes), frozenset(edges))
    return circuit


def export_blocks(
    sample_dir: Path,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    nodes: frozenset[Node],
    data_dir: Path,
    tokens: list[int],
    target_token_idx: int,
):
    """
    Create a JSON file with samples for every block in the circuit.
    """
    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get target feature magnitudes
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(input)

    # Get unique blocks from nodes
    blocks: set[tuple[int, int]] = set()
    for node in nodes:
        blocks.add((node.layer_idx, node.token_idx))

    # Get shard that was used for caching feature magnitudes
    shard = model_cache.get_shard(data_dir)

    # Export each block
    for layer_idx, token_idx in blocks:
        export_block(
            sample_dir,
            model,
            model_profile,
            model_cache,
            nodes,
            output,
            shard,
            layer_idx,
            token_idx,
            target_token_idx,
        )


def export_block(
    sample_dir: Path,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    nodes: frozenset[Node],
    output: SparsifiedGPTOutput,
    shard: DatasetShard,
    layer_idx: int,
    token_idx: int,
    target_token_idx: int,
):
    """
    Create a JSON file with samples for a specific block in the circuit.
    """
    target_feature_magnitudes = output.feature_magnitudes[layer_idx][0, token_idx, :].cpu().numpy()
    target_nodes = [n for n in nodes if n.token_idx == token_idx and n.layer_idx == layer_idx]
    circuit_feature_idxs = np.array([node.feature_idx for node in nodes if node in target_nodes])

    # Get samples that are similar to the target token
    cluster_search = ClusterSearch(model_profile, model_cache)
    cluster = cluster_search.get_cluster(
        layer_idx,
        token_idx,
        target_feature_magnitudes,
        circuit_feature_idxs,
        k_nearest=25,
        feature_coefficients=np.ones_like(circuit_feature_idxs),
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

    sample_dir.mkdir(parents=True, exist_ok=True)
    offset = target_token_idx - token_idx
    with open(sample_dir / f"{offset}.{layer_idx}.json", "w") as f:
        f.write(json_prettyprint(data))


def export_features(
    sample_dir,
    features_dir,
    nodes: frozenset[Node],
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    model_sample_set: ModelSampleSet,
    data_dir: Path,
    tokens: list[int],
    target_token_idx: int,
):
    """
    Create a JSON file with feature metrics for every feature in the circuit.
    """
    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get target feature magnitudes
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(input)

    # Get shard that was used for caching feature magnitudes
    shard = model_cache.get_shard(data_dir)

    # Export features with circuit context
    for node in nodes:
        layer_idx = node.layer_idx
        token_idx = node.token_idx
        feature_idx = node.feature_idx

        export_circuit_feature(
            sample_dir,
            model,
            model_profile,
            model_cache,
            output,
            shard,
            nodes,
            layer_idx,
            token_idx,
            feature_idx,
            target_token_idx,
        )

    # Export features without circuit context
    features: set[tuple[int, int]] = set()
    for node in nodes:
        features.add((node.layer_idx, node.feature_idx))
    sorted_features = sorted(list(features))
    for layer_idx, feature_idx in sorted_features:
        export_shared_feature(
            features_dir,
            model,
            model_profile,
            model_sample_set,
            shard,
            layer_idx,
            feature_idx,
        )


def export_shared_feature(
    features_dir,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_sample_set: ModelSampleSet,
    shard: DatasetShard,
    layer_idx: int,
    feature_idx: int,
):
    """
    Create a JSON file with feature metrics for a feature without any specific circuit context.
    """
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


def export_circuit_feature(
    sample_dir,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    output: SparsifiedGPTOutput,
    shard: DatasetShard,
    nodes: frozenset[Node],
    layer_idx: int,
    token_idx: int,
    feature_idx: int,
    target_token_idx: int,
):
    """
    Create a JSON file with feature metrics for a specific feature in the circuit.
    """
    # Load feature metrics
    feature_profile: FeatureProfile = model_profile[layer_idx][feature_idx]

    # Get target feature magnitudes
    target_feature_magnitudes = output.feature_magnitudes[layer_idx][0, token_idx, :].cpu().numpy()
    target_nodes = [n for n in nodes if n.token_idx == token_idx and n.layer_idx == layer_idx]
    circuit_feature_idxs = np.array([node.feature_idx for node in nodes if node in target_nodes])

    # Magnify the importance of the targeted feature
    feature_coefficients = np.full_like(circuit_feature_idxs, 0.04, dtype=np.float32)
    feature_coefficients[np.where(circuit_feature_idxs == feature_idx)[0]] = 1.0

    # Get samples that are similar to the target token
    num_samples = 25
    k_nearest = 25
    cluster_search = ClusterSearch(model_profile, model_cache)
    cluster = cluster_search.get_cluster(
        layer_idx,
        token_idx,
        target_feature_magnitudes,
        circuit_feature_idxs,
        # Use more neighbors if more than one feature dimension
        k_nearest=k_nearest,
        feature_coefficients=feature_coefficients,
        positional_coefficient=0.0,
    )

    # Choose random samples from the cluster
    sample_size = min(num_samples, len(cluster.idxs))
    sample_idxs = np.random.choice(cluster.idxs, size=sample_size, replace=False).tolist()

    # Pick samples from cluster
    samples: list[Sample] = []
    block_size = model_cache.block_size
    layer_cache = model_cache[layer_idx]
    for shard_token_idx in sample_idxs:  # type: ignore
        sample_block_idx = shard_token_idx // block_size
        sample_token_idx = shard_token_idx % block_size
        starting_idx = sample_block_idx * block_size
        ending_idx = starting_idx + block_size
        magnitudes = layer_cache.csr_matrix[starting_idx:ending_idx, feature_idx]  # Shape: (block_size, 1)
        magnitudes = magnitudes.transpose()  # Shape: (1, block_size)
        sample = Sample(
            layer_idx=layer_idx,
            block_idx=sample_block_idx,
            token_idx=sample_token_idx,
            magnitudes=magnitudes,
        )
        samples.append(sample)

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
    sample_dir.mkdir(parents=True, exist_ok=True)
    offset = target_token_idx - token_idx
    with open(sample_dir / f"{offset}.{layer_idx}.{feature_idx}.json", "w") as f:
        f.write(json_prettyprint(data))


def export_circuit_data(
    tokens: list[int],
    sample_dir: Path,
    model: SparsifiedGPT,
    model_profile: ModelProfile,
    model_cache: ModelCache,
    circuit: Circuit,
    edge_importance: dict[Edge, float],
    token_importance: dict[Node, dict[int, float]],
    target_token_idx: int,
    threshold: float,
):
    """
    Export sample data to data.json
    """
    # Convert tokens to tensor
    input: torch.Tensor = torch.tensor(tokens, device=model.config.device).unsqueeze(0)  # Shape: (1, T)

    # Get feature magnitudes
    with torch.no_grad():
        output: SparsifiedGPTOutput = model(input)

    # Data to export
    data = {
        "text": model.gpt.config.tokenizer.decode_sequence(tokens),
        "decodedTokens": [model.gpt.config.tokenizer.decode_token(token) for token in tokens],
        "targetIdx": target_token_idx,
        "kldThreshold": threshold,
    }

    # Set feature magnitudes
    data["activations"] = {}
    data["normalizedActivations"] = {}
    for node in circuit.nodes:
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
    data["circuitProbabilities"] = {k: round(v / 100.0, 3) for k, v in predicted_probabilities.items() if v > 0.1}

    # Set ablation graph
    data["graph"] = {}
    for downstream_node in sorted(set(edge.downstream for edge in circuit.edges)):
        dependencies = []
        upstream_edges = [edge for edge in circuit.edges if edge.downstream == downstream_node]
        for edge in sorted(upstream_edges):
            edge_weight = edge_importance[edge]
            dependencies.append([node_to_key(edge.upstream, target_token_idx), edge_weight])
        data["graph"][node_to_key(downstream_node, target_token_idx)] = dependencies

    # Set block importance
    data["blockImportance"] = {}
    for downstream_node in sorted(set(edge.downstream for edge in circuit.edges)):
        groups = []
        upstream_edges = [edge for edge in circuit.edges if edge.downstream == downstream_node]
        upstream_blocks = set((edge.upstream.layer_idx, edge.upstream.token_idx) for edge in upstream_edges)
        for layer_idx, token_idx in sorted(upstream_blocks):
            block_weight = token_importance[downstream_node].get(token_idx, 0.0)
            groups.append([f"{target_token_idx - token_idx}.{layer_idx}", block_weight])
        data["blockImportance"][node_to_key(downstream_node, target_token_idx)] = groups

    # Export to data.json
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "data.json", "w") as f:
        f.write(json_prettyprint(data))


def node_to_key(node: Node, target_token_idx: int) -> str:
    """
    Convert a node to a string key.
    """
    token_offset = target_token_idx - node.token_idx
    return f"{token_offset}.{node.layer_idx}.{node.feature_idx}"


if __name__ == "__main__":
    main()
