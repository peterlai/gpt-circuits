import math
from collections import defaultdict
from typing import Optional

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
    Search for circuit edges in a sparsified model.
    """

    def __init__(self, model: SparsifiedGPT, model_profile: ModelProfile, ablator: Ablator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param model_profile: The model profile containing cache feature metrics.
        :param ablator: Ablation tecnique to use for circuit extraction.
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
        threshold: float,
    ) -> frozenset[Edge]:
        """
        Search for circuit edges in a sparsified model.

        :param tokens: The token sequence to use for circuit extraction.
        :param upstream_nodes: The upstream nodes to use for circuit extraction.
        :param downstream_nodes: The downstream nodes to use for circuit extraction.
        :param threshold: Mean-squared increase to use as search threshold (e.g. 0.1 = 10% increase).
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

        # Set initial edges as all edges that could exist between upstream and downstream nodes
        initial_edges = set()
        for upstream in sorted(upstream_nodes):
            for downstream in sorted(downstream_nodes):
                if upstream.token_idx <= downstream.token_idx:
                    initial_edges.add(Edge(upstream, downstream))

        # Set baseline MSE to use for computing search threshold
        baseline_error = self.estimate_downstream_mse(
            downstream_nodes,
            frozenset(initial_edges),
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )
        print(f"MSE baseline: {baseline_error:.4f}")

        # Starting search states
        search_threshold = baseline_error * (threshold + 1.0)  # Threshold parameter defines MSE increase fraction
        circuit_edges: frozenset[Edge] = frozenset(initial_edges)  # Circuit to start pruning
        discard_candidates: frozenset[Edge] = frozenset()
        downstream_error: float = float("inf")
        print(f"MSE threshold: {search_threshold:.4f}")

        # Start search
        for _ in range(len(initial_edges)):
            # Derive downstream magnitudes from upstream magnitudes and edges to produce a mean-squared error
            circuit_candidate = Circuit(downstream_nodes, edges=frozenset(circuit_edges - discard_candidates))
            downstream_error = self.estimate_downstream_mse(
                downstream_nodes,
                circuit_candidate.edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )

            # Print results
            print(
                f"Edges: {len(circuit_candidate.edges)}/{len(initial_edges)} - "
                f"Downstream MSE: {downstream_error:.4f}"
            )

            # If below threshold, continue search
            if downstream_error < search_threshold:
                # Update circuit
                circuit_edges = circuit_candidate.edges

                # Find least important edges
                if least_important_edges := self.find_least_important_edges(
                    downstream_nodes,
                    circuit_edges,
                    upstream_magnitudes,
                    original_downstream_magnitudes,
                    target_token_idx,
                    # Remove 4% of edges on each iteration
                    max_count=int(math.ceil(len(circuit_edges) * 0.04)),
                ):
                    discard_candidates = frozenset(least_important_edges)
                else:
                    print("Stopping search - No more edges can be removed.")
                    break

            # If above threshold, stop search
            else:
                print("Stopping search - Downstream error is too high.")
                break

        # Print final edges (grouped by downstream token)
        print(f"\nCircuit after edge search ({len(circuit_edges)}):")
        for downstream_node in sorted(downstream_nodes):
            edges = [edge for edge in circuit_edges if edge.downstream == downstream_node]
            print(f"{downstream_node}: {', '.join([str(edge.upstream) for edge in sorted(edges)])}")

        return circuit_edges

    def find_least_important_edges(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
        max_count: int = 1,
    ) -> Optional[set[Edge]]:
        """
        Find the least important edge in a circuit. Returns the edge and its mean-squared error.
        To avoid having unconnected nodes, the last edge to a node will not be considered.

        :param max_count: The maximum number of edges to return.
        """
        # Map edges to ablation effects
        edge_to_error = self.estimate_edge_ablation_effects(
            downstream_nodes,
            edges,
            upstream_magnitudes,
            original_downstream_magnitudes,
            target_token_idx,
        )

        # Sort edges by mean-squared error (ascending)
        sorted_edges = sorted(edge_to_error.items(), key=lambda x: x[1])

        # Edges to be discarded
        discard_candidates: set[Edge] = set()

        # Ignore edges that would leave a node unconnected
        for edge, _ in sorted_edges:
            if len([e for e in edges if e.downstream == edge.downstream]) > 1:
                if len([e for e in edges if e.upstream == edge.upstream]) > 1:
                    discard_candidates.add(edge)
                    if len(discard_candidates) >= max_count:
                        break

        # Return least important edges
        if discard_candidates:
            return discard_candidates

        # If no discard candidates, all edges are required
        return None

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

        # Create a set of circuit variants with one edge removed
        circuit_variants: list[Circuit] = []
        edge_to_circuit_variant: dict[Edge, Circuit] = {}
        for edge in edges:
            circuit_variant = Circuit(downstream_nodes, edges=frozenset(edges - {edge}))
            circuit_variants.append(circuit_variant)
            edge_to_circuit_variant[edge] = circuit_variant

        # Compute downstream feature magnitude errors for each circuit variant
        downstream_errors: dict[Circuit, float] = {}
        for circuit_variant in circuit_variants:
            error = self.estimate_downstream_mse(
                downstream_nodes,
                circuit_variant.edges,
                upstream_magnitudes,
                original_downstream_magnitudes,
                target_token_idx,
            )
            downstream_errors[circuit_variant] = error

        # Map edges to mean-squared errors
        return {edge: downstream_errors[variant] for edge, variant in edge_to_circuit_variant.items()}

    def estimate_downstream_mse(
        self,
        downstream_nodes: frozenset[Node],
        edges: frozenset[Edge],
        upstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        original_downstream_magnitudes: torch.Tensor,  # Shape: (T, F)
        target_token_idx: int,
    ) -> float:
        """
        Use downstream feature magnitudes derived from upstream feature magnitudes and edges to produce a mean-squared error.

        :param downstream_nodes: The downstream nodes to use for deriving downstream feature magnitudes.
        :param edges: The edges to use for deriving downstream feature magnitudes.
        :param upstream_magnitudes: The upstream feature magnitudes.
        :param original_downstream_magnitudes: The original downstream feature magnitudes.
        :param target_token_idx: The target token index.
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
        downstream_mses = torch.zeros(len(downstream_nodes))
        for i, (node, sampled_magnitudes) in enumerate(node_to_sampled_magnitudes.items()):
            original_magnitude = original_downstream_magnitudes[node.token_idx, node.feature_idx]
            downstream_mses[i] = torch.mean((norm_coefficients[i] * (sampled_magnitudes - original_magnitude)) ** 2)

        return downstream_mses.mean().item()
