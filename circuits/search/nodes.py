import math
from dataclasses import dataclass

import torch

from circuits import Circuit, Node, SearchConfiguration
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import (
    analyze_circuit_divergence,
    analyze_token_mask_divergence,
)
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

    def __init__(self, model: SparsifiedGPT, ablator: ResampleAblator, config: SearchConfiguration):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param ablator: Ablation tecnique to use for circuit extraction.
        :param config: Configuration for the search.
        """
        self.model = model
        self.ablator = ablator
        self.config = config

    def search(
        self,
        tokens: list[int],
        upstream_nodes: frozenset[Node],
        layer_idx: int,
        target_token_idx: int,
    ) -> frozenset[RankedNode]:
        """
        Search for circuit nodes in the selected layer.

        :param model_output: The sparsified model output.
        :param upstream_nodes: The upstream nodes in the circuit.
        :param layer_idx: The layer index to search in.
        :param target_token_idx: The target token index.
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
        basline_results = analyze_circuit_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [baseline_circuit],
            feature_magnitudes,
            num_samples=self.config.num_node_samples,
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
            max_count=self.config.max_token_positions,  # Limit the number of token indices to consider
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

        # Filter ranked nodes
        selected_nodes = self.filter_ranked_nodes(ranked_nodes, self.config.stoppage_window, self.config.threshold)
        print(f"Added the following nodes to layer {layer_idx}: {[rn.node for rn in selected_nodes]}")
        return selected_nodes

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
            circuit_analysis = analyze_circuit_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit],
                feature_magnitudes,
                num_samples=self.config.num_node_samples,
            )[circuit]

            # Find discard candidates
            discard_candidates = self.find_least_important_nodes(
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                layer_nodes=layer_nodes,
                # Remove 8% of nodes on each iteration
                max_count=int(math.ceil(len(layer_nodes) * 0.08)),
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

    def filter_ranked_nodes(
        self, ranked_nodes: list[RankedNode], stoppage_window: int | None, threshold: float
    ) -> frozenset[RankedNode]:
        """
        Walk backwards through ranked nodes and stop when KL divergence is below search threshold.
        NOTE: KL divergence does not monotonically decrease, which is why we need to walk backwards.
        """
        selected_nodes: set[RankedNode] = set()
        previous_klds = []
        for ranked_node in reversed(ranked_nodes):
            # Stop if KLD is greater than average of previous few values
            if w := stoppage_window:
                if len(previous_klds) >= w and ranked_node.kld > sum(previous_klds[-w:]) / w:
                    break
            # Add node to layer
            selected_nodes.add(ranked_node)
            # Stop if KLD is below search threshold
            if ranked_node.kld < threshold:
                break
            previous_klds.append(ranked_node.kld)

        return frozenset(selected_nodes)

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
        kld_results = analyze_circuit_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [variant for variant in circuit_variants.values()],
            feature_magnitudes,
            self.config.num_node_samples,
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

                # Create a token mask from selected token idxs
                token_mask = torch.zeros(size=(feature_magnitudes.shape[0],), dtype=torch.bool)
                token_mask[list(selected_token_idxs)] = True

                # Compute KL divergence
                analysis = analyze_token_mask_divergence(
                    self.model,
                    self.ablator,
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    token_mask,
                    feature_magnitudes,
                    num_samples=self.config.num_node_samples,
                )

                # Print results
                print(
                    f"Tokens: {len(selected_token_idxs)}/{target_token_idx + 1} - "
                    f"KL Div: {analysis.kl_divergence:.4f} - "
                    f"Added: {set([n.token_idx for n in new_nodes]) or 'None'} - "
                    f"Predictions: {analysis.predictions}"
                )

                # If below threshold, stop search
                # NOTE: Using lower threshold for coarse token search
                if analysis.kl_divergence < self.config.threshold * 0.75:
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
        token_mask_variants: dict[int, torch.Tensor] = {}
        unique_token_indices = {node.token_idx for node in remaining_nodes}
        for token_idx in unique_token_indices:
            token_mask = torch.zeros(size=(feature_magnitudes.shape[0],), dtype=torch.bool)
            token_mask[list({node.token_idx for node in circuit_nodes} | {token_idx})] = True
            token_mask_variants[token_idx] = token_mask

        # Map token indices to KL divergence
        results = {}
        for token_idx, token_mask in token_mask_variants.items():
            kld_results = analyze_token_mask_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                token_mask,
                feature_magnitudes,
                # Increase number of samples for token selection
                min(self.config.num_node_samples * 4, self.config.k_nearest or int(1e10)),
            )
            results[token_idx] = kld_results.kl_divergence

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
