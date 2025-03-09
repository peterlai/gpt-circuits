import math
from dataclasses import dataclass
from typing import Sequence

import torch

from circuits import Circuit, Node
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import analyze_divergence, get_predictions
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


@dataclass(frozen=True)
class RankedNode:
    """
    A node and its importance.
    """

    node: Node
    rank: int  # Node importance
    kld: float  # KLD ceiling associated with this node


class NodeSearch:
    """
    Search for circuit nodes in a sparsified model.
    """

    def __init__(self, model: SparsifiedGPT, ablator: ResampleAblator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param ablator: Ablation tecnique to use for circuit extraction.
        :param num_samples: The number of samples to use for ablation.
        """
        self.model = model
        self.ablator = ablator
        self.num_samples = num_samples

    def search(
        self,
        tokens: list[int],
        upstream_nodes: frozenset[Node],
        layer_idx: int,
        target_token_idx: int,
        threshold: float,
    ) -> frozenset[RankedNode]:
        """
        Search for circuit nodes in the selected layer.

        :param model_output: The sparsified model output.
        :param upstream_nodes: The upstream nodes in the circuit.
        :param layer_idx: The layer index to search in.
        :param target_token_idx: The target token index.
        :param threshold: The KL diverence threshold for node extraction.
        """
        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get target logits
        with torch.no_grad():
            model_output: SparsifiedGPTOutput = self.model(input)

        # Get output for layer
        feature_magnitudes = model_output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)
        target_logits = model_output.logits.squeeze(0)[target_token_idx]  # Shape: (V)

        # Set starting token index
        if layer_idx == self.num_layers - 1:
            # In the last layer, we only care about the target token.
            start_token_idx = target_token_idx
        else:
            # Tokens before the leftmost upstream node are not important.
            start_token_idx = min([node.token_idx for node in upstream_nodes] or [0])

        # Get non-zero features where token index is in [start_token_idx...target_token_idx]
        layer_nodes = set({})
        non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
        for t, f in zip(*non_zero_indices):
            if t >= start_token_idx and t <= target_token_idx:
                layer_nodes.add(Node(layer_idx, t.item(), f.item()))
        layer_nodes = frozenset(layer_nodes)

        # Get KLD baseline
        baseline_circuit = Circuit(layer_nodes)
        basline_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [baseline_circuit],
            feature_magnitudes,
            num_samples=self.num_samples * 4,  # Increase samples for baseline
        )[baseline_circuit]
        baseline_kld = basline_results.kl_divergence

        # Narrow down nodes to consider by selecting token indices that are most likely to be important
        print(f"\nSelecting tokens in layer {layer_idx} with important information...")
        circuit_candidates = self.select_nodes_by_token_idx(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            layer_nodes,
            upstream_nodes,
            threshold=threshold / 2,  # Use lower threshold for coarse search
            max_count=16,  # Limit to 16 token indices
        )

        # Rank remaining features by importance
        print(f"\nRanking nodes in layer {layer_idx} (baseline KLD: {baseline_kld:.4f})...")
        ranked_nodes = self.rank_nodes(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            circuit_candidates,
        )

        # Walk backwards through ranked nodes and stop when KL divergence is below search threshold.
        # NOTE: KL divergence does not monotonically decrease, which is why we need to walk backwards.
        selected_nodes: set[RankedNode] = set()
        previous_klds = []
        for ranked_node in reversed(ranked_nodes):
            # Stop if KLD is greater than average of previous 3 values
            if len(previous_klds) >= 3 and ranked_node.kld > sum(previous_klds[-3:]) / 3:
                break
            # Add node to layer
            selected_nodes.add(ranked_node)
            # Stop if KLD is below search threshold
            if ranked_node.kld < threshold:
                break
            previous_klds.append(ranked_node.kld)

        print(f"Added the following nodes to layer {layer_idx}: {[rn.node for rn in selected_nodes]}")
        return frozenset(selected_nodes)

    def rank_nodes(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
    ) -> list[RankedNode]:
        """
        Rank the nodes in a layer from least to most important.
        """
        # Nodes ordered by when they were removed from the circuit
        ranked_nodes: list[RankedNode] = []

        # Starting search states
        initial_nodes = layer_nodes
        discard_candidates: set[Node] = set({})

        # Remove nodes from circuit until empty
        while layer_nodes:
            # Compute KL divergence
            circuit = Circuit(nodes=layer_nodes)
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit],
                feature_magnitudes,
                num_samples=self.num_samples,
            )[circuit]

            # Find discard candidates
            discard_candidates = self.find_least_important_nodes(
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                layer_nodes=layer_nodes,
                # Remove 4% of nodes on each iteration
                max_count=int(math.ceil(len(layer_nodes) * 0.04)),
            )

            # Map discard candidates to current KL divergence
            for node in discard_candidates:
                ranked_nodes.append(RankedNode(node, len(layer_nodes), circuit_analysis.kl_divergence))

            # Print results
            print(
                f"Features: {len(layer_nodes)}/{len(initial_nodes)} - "
                f"KL Div: {circuit_analysis.kl_divergence:.4f} - "
                f"Nodes: {set([(n.token_idx, n.feature_idx) for n in discard_candidates])} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # Update circuit
            layer_nodes = frozenset(layer_nodes - discard_candidates)

        return ranked_nodes

    def find_least_important_nodes(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
        max_count: int,
    ) -> set[Node]:
        """
        Return the least important nodes in a layer.
        """
        # Generate all circuit variations with one node removed
        circuit_variants: dict[Node, Circuit] = {}
        for node in layer_nodes:
            circuit_variants[node] = Circuit(nodes=frozenset([n for n in layer_nodes if n != node]))

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [variant for variant in circuit_variants.values()],
            feature_magnitudes,
            self.num_samples,
        )

        # Map nodes to KL divergence
        node_to_kld = {node: kld_results[variant].kl_divergence for node, variant in circuit_variants.items()}

        # Find least important nodes
        sorted_nodes = sorted(node_to_kld.items(), key=lambda x: x[1])  # Sort by KL divergence (ascending)
        least_important_nodes = set([node for node, _ in sorted_nodes[:max_count]])
        return least_important_nodes

    def select_nodes_by_token_idx(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
        upstream_nodes: frozenset[Node],
        threshold: float,
        max_count: int,
    ) -> frozenset[Node]:
        """
        Select tokens that are most likely to be important.
        """
        # Group features by token index
        nodes_by_token_idx: dict[int, set[Node]] = {}
        for token_idx in range(target_token_idx + 1):
            nodes_by_token_idx[token_idx] = set({node for node in layer_nodes if node.token_idx == token_idx})

        # Select nodes that share token indices with upstream nodes and include nodes at the target index.
        selected_nodes: frozenset[Node] = frozenset(nodes_by_token_idx[target_token_idx])
        for downstream_idx in {node.token_idx for node in upstream_nodes}:
            selected_nodes = frozenset(selected_nodes | nodes_by_token_idx[downstream_idx])

        # If no upstream nodes, start search for new token indices.
        if not upstream_nodes:
            new_nodes: set[Node] = set({})

            # Start search
            while selected_nodes is not layer_nodes:
                # Update circuit
                selected_nodes = frozenset(selected_nodes | new_nodes)
                selected_token_idxs = {node.token_idx for node in selected_nodes}

                # Compute KL divergence
                circuit = Circuit(nodes=selected_nodes)
                circuit_analysis = analyze_divergence(
                    self.model,
                    self.ablator,
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    [circuit],
                    feature_magnitudes,
                    num_samples=self.num_samples,
                )[circuit]

                # Print results
                print(
                    f"Tokens: {len(set(f.token_idx for f in circuit.nodes))}/{target_token_idx + 1} - "
                    f"KL Div: {circuit_analysis.kl_divergence:.4f} - "
                    f"Added: {set([n.token_idx for n in new_nodes]) or 'None'} - "
                    f"Predictions: {circuit_analysis.predictions}"
                )

                # If below threshold, stop search
                if circuit_analysis.kl_divergence < threshold:
                    print("Reached target KL divergence.")
                    break

                # Stop early if reached the max number of token indices
                if len(selected_token_idxs) >= max_count:
                    print("Reached token limit.")
                    break

                # If no remaining nodes, stop search
                remaining_nodes = frozenset(layer_nodes - selected_nodes)
                if not remaining_nodes:
                    print("No remaining nodes to add.")
                    break

                # Find next most important token
                most_important_token_idx = self.find_next_token(
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    feature_magnitudes,
                    circuit_nodes=selected_nodes,
                    remaining_nodes=layer_nodes - selected_nodes,
                )
                new_nodes = nodes_by_token_idx[most_important_token_idx]

        # Return circuit
        print(f"Selected token indices: {','.join(map(str, sorted({n.token_idx for n in selected_nodes})))}")
        return selected_nodes

    def find_next_token(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_nodes: frozenset[Node],
        remaining_nodes: frozenset[Node],
    ) -> int:
        """
        Find next token to add to the circuit.
        """
        # Generate all variations with one token added
        circuit_variants: dict[int, Circuit] = {}
        unique_token_indices = {node.token_idx for node in remaining_nodes}
        for token_idx in unique_token_indices:
            new_nodes = frozenset([node for node in remaining_nodes if node.token_idx == token_idx])
            circuit_variant = Circuit(nodes=frozenset(circuit_nodes | new_nodes))
            circuit_variants[token_idx] = circuit_variant

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [variant for variant in circuit_variants.values()],
            feature_magnitudes,
            self.num_samples * 4,  # Increase number of samples for token selection
        )

        # Map token indices to KL divergence
        results = {token_idx: kld_results[variant].kl_divergence for token_idx, variant in circuit_variants.items()}

        # The most important token is the one that results in the lowest KL divergence
        sorted_tokens = sorted(results.items(), key=lambda x: x[1])
        most_important_token_idx = sorted_tokens[0][0]
        return most_important_token_idx

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1  # Add 1 for the embedding layer
