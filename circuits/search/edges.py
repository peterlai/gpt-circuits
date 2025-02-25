from collections import defaultdict

import torch

from circuits import Circuit, Edge, Node
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import Ablator
from circuits.search.divergence import (
    compute_downstream_magnitudes,
    patch_feature_magnitudes,
)
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class EdgeSearch:
    """
    Analyze edge importance in a circuit by ablating each edge between two adjacent layers.
    """

    def __init__(self, model: SparsifiedGPT, model_profile: ModelProfile, ablator: Ablator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit analysis.
        :param model_profile: The model profile containing cache feature metrics.
        :param ablator: Ablation tecnique to use for circuit analysis.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.model_profile = model_profile
        self.ablator = ablator
        self.num_samples = num_samples

    def search(
        self,
        tokens: list[int],
        target_token_idx: int,
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
    ) -> dict[Edge, float]:
        """
        Map each edge in a sparsified model to a normalized MSE increase that results from its ablation.

        :param tokens: The token sequence to use for circuit extraction.
        :param upstream_nodes: The upstream nodes to use for circuit analysis.
        :param downstream_nodes: The downstream nodes to use for circuit analsysi.
        """
        assert len(downstream_nodes) > 0
        downstream_idx = next(iter(downstream_nodes)).layer_idx
        upstream_idx = downstream_idx - 1

        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get feature magnitudes
        with torch.no_grad():
            output: SparsifiedGPTOutput = self.model(input)
        upstream_magnitudes = output.feature_magnitudes[upstream_idx].squeeze(0)  # Shape: (T, F)
        original_downstream_magnitudes = output.feature_magnitudes[downstream_idx].squeeze(0)  # Shape: (T, F)

        # Find all edges that could exist between upstream and downstream nodes
        all_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    all_edges.add(Edge(upstream, downstream))

        # Set baseline MSE to use for comparisons
        baseline_mses = self.estimate_downstream_mses(
            downstream_nodes,
            frozenset(all_edges),
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Map edges to ablation effects
        ablation_mses = self.estimate_edge_ablation_effects(
            downstream_nodes,
            frozenset(all_edges),
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Calculate MSE increase from baseline
        edge_importance = {}
        for edge, mse in sorted(ablation_mses.items(), key=lambda x: x[0]):
            baseline_mse = baseline_mses[edge.downstream]
            print(f"Edge {edge} - Baseline MSE: {baseline_mse:.4f} - Ablation MSE: {mse:.4f}")
            edge_importance[edge] = (mse - baseline_mse) / baseline_mse  # normalized MSE increase

        return edge_importance

    def estimate_edge_ablation_effects(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Edge, float]:
        """
        Estimate the downstream feature mean-squared error that results from ablating each edge in a circuit.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        """
        # Maps edge to downstream mean-squared error
        edge_to_mse: dict[Edge, float] = {}

        # Create a set of circuit variants with one edge removed
        edge_to_circuit_variant: dict[Edge, Circuit] = {}
        for edge in edges:
            circuit_variant = Circuit(downstream_nodes, edges=frozenset(edges - {edge}))
            edge_to_circuit_variant[edge] = circuit_variant

        # Compute downstream feature magnitude errors that results from ablating each edge
        for edge, circuit_variant in edge_to_circuit_variant.items():
            downstream_errors = self.estimate_downstream_mses(
                downstream_nodes,
                circuit_variant.edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )
            edge_to_mse[edge] = downstream_errors[edge.downstream]
        return edge_to_mse

    def estimate_downstream_mses(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Node, float]:
        """
        Use downstream feature magnitudes derived from upstream feature magnitudes and edges to produce a mean-squared
        error per downstream node.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        :param target_token_idx: The target token index.

        :return: The mean-squared error per downstream node.
        """
        # Map downstream nodes to upstream dependencies
        node_to_dependencies: dict[Node, frozenset[Node]] = {}
        for node in downstream_nodes:
            node_to_dependencies[node] = frozenset([edge.upstream for edge in edges if edge.downstream == node])
        dependencies_to_nodes: dict[frozenset[Node], set[Node]] = defaultdict(set)
        for node, dependencies in node_to_dependencies.items():
            dependencies_to_nodes[dependencies].add(node)

        # Patch upstream feature magnitudes for each set of dependencies
        circuit_variants = [Circuit(nodes=dependencies) for dependencies in dependencies_to_nodes.keys()]
        upstream_layer_idx = next(iter(downstream_nodes)).layer_idx - 1
        patched_upstream_magnitudes = patch_feature_magnitudes(  # Shape: (num_samples, T, F)
            self.ablator,
            upstream_layer_idx,
            target_token_idx,
            circuit_variants,
            upstream_magnitudes,
            num_samples=self.num_samples,
        )

        # Compute downstream feature magnitudes for each set of dependencies
        sampled_downstream_magnitudes = compute_downstream_magnitudes(  # Shape: (num_samples, T, F)
            self.model,
            upstream_layer_idx,
            patched_upstream_magnitudes,
        )

        # Map each downstream node to a set of sampled feature magnitudes
        node_to_sampled_magnitudes: dict[Node, torch.Tensor] = {}
        for circuit_variant, magnitudes in sampled_downstream_magnitudes.items():
            for node in dependencies_to_nodes[circuit_variant.nodes]:
                node_to_sampled_magnitudes[node] = magnitudes[:, node.token_idx, node.feature_idx]

        # Caculate normalization coefficients for downstream features, which scale magnitudes to [0, 1]
        norm_coefficients = torch.ones(len(downstream_nodes))
        layer_profile = self.model_profile[upstream_layer_idx + 1]
        for i, node in enumerate(node_to_sampled_magnitudes.keys()):
            feature_profile = layer_profile[int(node.feature_idx)]
            norm_coefficients[i] = 1.0 / feature_profile.max

        # Calculate mean-squared error from original downstream feature magnitudes
        downstream_mses = {}
        for node, sampled_magnitudes in node_to_sampled_magnitudes.items():
            original_magnitude = original_downstream_magnitudes[node.token_idx, node.feature_idx]
            normalized_mse = torch.mean((norm_coefficients[i] * (sampled_magnitudes - original_magnitude)) ** 2)
            downstream_mses[node] = normalized_mse.item()
        return downstream_mses
