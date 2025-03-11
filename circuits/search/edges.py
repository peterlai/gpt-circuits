from collections import defaultdict
from dataclasses import dataclass

import torch

from circuits import Circuit, Edge, Node
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import (
    compute_downstream_magnitudes,
    patch_feature_magnitudes,
)
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


@dataclass(frozen=True)
class EdgeSearchResult:
    """
    Result of an edge search.
    """

    # Maps an edge to a normalized MSE increase
    edge_importance: dict[Edge, float]

    # Maps a downstream node to the normalized MSE increase that results from ablating all features at each upstream
    # token index.
    token_importance: dict[Node, dict[int, float]]


class EdgeSearch:
    """
    Analyze edge importance in a circuit by ablating each edge between two adjacent layers.
    """

    def __init__(
        self,
        model: SparsifiedGPT,
        model_profile: ModelProfile,
        upstream_ablator: ResampleAblator,
        num_samples: int,
    ):
        """
        :param model: The sparsified model to use for circuit analysis.
        :param model_profile: The model profile containing cache feature metrics.
        :param upstream_ablator: Ablator to use for patching upstream feature magnitudes.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.model_profile = model_profile
        self.upstream_ablator = upstream_ablator
        self.num_samples = num_samples

    def search(
        self,
        tokens: list[int],
        upstream_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
        target_token_idx: int,
    ) -> EdgeSearchResult:
        """
        Map each edge in a sparsified model to a normalized MSE increase that results from its ablation.
        """
        assert len(downstream_nodes) > 0
        downstream_idx = next(iter(downstream_nodes)).layer_idx
        upstream_idx = downstream_idx - 1
        print(f"\nAnalyzing edge importance between layers {upstream_idx} and {downstream_idx}...")

        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get feature magnitudes
        with torch.no_grad():
            model_output: SparsifiedGPTOutput = self.model(input)

        upstream_magnitudes = model_output.feature_magnitudes[upstream_idx].squeeze(0)  # Shape: (T, F)
        original_downstream_magnitudes = model_output.feature_magnitudes[downstream_idx].squeeze(0)  # Shape: (T, F)

        # Find all edges that could exist between upstream and downstream nodes
        all_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    all_edges.add(Edge(upstream, downstream))
        all_edges = frozenset(all_edges)

        # Set baseline MSE to use for comparisons
        baseline_mses = self.estimate_downstream_mses(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Compute edge importance
        edge_importance = self.compute_edge_importance(
            all_edges,
            downstream_nodes,
            baseline_mses,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Compute token importance
        token_importance = self.compute_token_importance(
            all_edges,
            downstream_nodes,
            baseline_mses,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        return EdgeSearchResult(edge_importance=edge_importance, token_importance=token_importance)

    def compute_edge_importance(
        self,
        all_edges: frozenset[Edge],
        downstream_nodes: frozenset[Node],
        baseline_mses: dict[Node, float],
        upstream_magnitudes: torch.Tensor,
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Edge, float]:
        """
        Compute the importance of edges between upstream and downstream nodes.

        :param all_edges: Set of all possible edges between layers
        :param downstream_nodes: Set of downstream nodes
        :param baseline_mses: The baseline mean-squared error per downstream node
        :param upstream_magnitudes: The upstream feature magnitudes (shape: T, F)
        :param original_downstream_magnitudes: The original downstream feature magnitudes (shape: T, F)
        :param target_token_idx: The target token index

        :return: Dictionary mapping edges to their importance scores
        """
        # Map edges to ablation effects
        ablation_mses = self.estimate_edge_ablation_effects(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Calculate MSE increase from baseline
        edge_mse_increase = {}
        for edge, mse in sorted(ablation_mses.items(), key=lambda x: x[0]):
            baseline_mse = baseline_mses[edge.downstream]
            edge_mse_increase[edge] = mse - baseline_mse

        # Calculate MSE increase stats per downstream node
        min_mse_increases: dict[Node, float] = {}
        max_mse_increases: dict[Node, float] = {}
        for downstream_node in downstream_nodes:
            upstream_edges = {edge for edge in edge_mse_increase.keys() if edge.downstream == downstream_node}
            mse_increases = [edge_mse_increase[edge] for edge in upstream_edges]
            min_mse_increase = min(mse_increases, default=0)
            min_mse_increases[downstream_node] = min_mse_increase
            max_mse_increase = max(mse_increases, default=0)
            max_mse_increases[downstream_node] = max_mse_increase

        # Print MSE increase stats
        for downstream_node in sorted(downstream_nodes):
            print(
                f"Edges from {downstream_node} - "
                f"Baseline: {baseline_mses[downstream_node]:.4f} - "
                f"Min MSE increase: {min_mse_increases[downstream_node]:.4f} - "
                f"Max MSE increase: {max_mse_increases[downstream_node]:.4f}"
            )

        # Normalize MSE increase by max MSE increase
        edge_importance = {}
        for edge, mse_increase in edge_mse_increase.items():
            mse_increase = max(mse_increase, 0)  # Avoid negative values
            max_mse_increase = max(max_mse_increases[edge.downstream], 1e-6)  # Avoid negative values
            edge_importance[edge] = mse_increase / max_mse_increase

        return edge_importance

    def compute_token_importance(
        self,
        all_edges: frozenset[Edge],
        downstream_nodes: frozenset[Node],
        baseline_mses: dict[Node, float],
        upstream_magnitudes: torch.Tensor,
        original_downstream_magnitudes: torch.Tensor,
        target_token_idx: int,
    ) -> dict[Node, dict[int, float]]:
        """
        Compute the importance of upstream tokens for downstream nodes.

        :param all_edges: Set of all possible edges between layers
        :param downstream_nodes: Set of downstream nodes
        :param baseline_mses: The baseline mean-squared error per downstream node
        :param upstream_magnitudes: The upstream feature magnitudes (shape: T, F)
        :param original_downstream_magnitudes: The original downstream feature magnitudes (shape: T, F)
        :param target_token_idx: The target token index

        :return: Dictionary mapping downstream nodes to dictionaries of token indices and their importance scores
        """
        # For each downstream node, map upstream token indices to an MSE
        token_ablation_mses = self.estimate_token_ablation_effects(
            downstream_nodes,
            all_edges,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Calculate token MSE increase from baseline
        token_mse_increases = defaultdict(dict)
        for downstream_node, upstream_token_mses in sorted(token_ablation_mses.items(), key=lambda x: x[0]):
            for token_idx, token_mse in sorted(upstream_token_mses.items(), key=lambda x: x[0]):
                baseline_mse = baseline_mses[downstream_node]
                token_mse_increases[downstream_node][token_idx] = token_mse - baseline_mse

        # Calculate MSE increase stats per downstream node
        min_mse_increases: dict[Node, float] = {}
        max_mse_increases: dict[Node, float] = {}
        for downstream_node in downstream_nodes:
            min_mse_increase = min(token_mse_increases[downstream_node].values(), default=0)
            min_mse_increases[downstream_node] = min_mse_increase
            max_mse_increase = max(token_mse_increases[downstream_node].values(), default=0)
            max_mse_increases[downstream_node] = max_mse_increase

        # Print MSE increase stats
        for downstream_node in sorted(downstream_nodes):
            print(
                f"Tokens from {downstream_node} - "
                f"Baseline: {baseline_mses[downstream_node]:.4f} - "
                f"Min MSE increase: {min_mse_increases[downstream_node]:.4f} - "
                f"Max MSE increase: {max_mse_increases[downstream_node]:.4f}"
            )

        # Normalize MSE increase by max MSE increase
        token_importance = defaultdict(dict)
        for downstream_node, token_mses in token_mse_increases.items():
            for token_idx, mse_increase in token_mses.items():
                mse_increase = max(mse_increase, 0)  # Avoid negative values
                max_mse_increase = max(max_mse_increases[downstream_node], 1e-6)  # Avoid negative value
                token_importance[downstream_node][token_idx] = mse_increase / max_mse_increase

        return token_importance

    def estimate_token_ablation_effects(
        self,
        downstream_nodes: frozenset[Node],
        all_edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> dict[Node, dict[int, float]]:
        """
        Estimate the downstream feature mean-squared error that results from ablating each token in a circuit.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        :param target_token_idx: The target token index.
        """
        token_ablation_mses = defaultdict(dict)
        for token_idx in sorted({edge.upstream.token_idx for edge in all_edges}):
            # Exclude edges that are connected to the target token
            patched_edges = frozenset({edge for edge in all_edges if edge.upstream.token_idx != token_idx})
            estimated_mses = self.estimate_downstream_mses(
                downstream_nodes,
                patched_edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )
            # Look for downstream nodes that have an edge to the target token
            for downstream_node in {edge.downstream for edge in all_edges if edge.upstream.token_idx == token_idx}:
                # Set the mean-squared error from the downstream node
                token_ablation_mses[downstream_node][token_idx] = estimated_mses[downstream_node]

        return token_ablation_mses

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
        :param target_token_idx: The target token index.
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
            self.upstream_ablator,
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

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1  # Add 1 for the embedding layer
