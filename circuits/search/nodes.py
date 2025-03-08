import math
from typing import Sequence

import torch

from circuits import Circuit, Node
from circuits.search.ablation import Ablator
from circuits.search.divergence import analyze_divergence, get_predictions
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class NodeSearch:
    """
    Search for circuit nodes in a sparsified model.
    """

    def __init__(self, model: SparsifiedGPT, ablator: Ablator, num_samples: int):
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
        target_token_idx: int,
        threshold: float,
    ) -> tuple[frozenset[Node], dict[int, float], dict[int, dict]]:
        """
        Find the nodes in a circuit.

        :return: A tuple containing the following:
            - A frozenset of nodes in the circuit.
            - A dictionary mapping layer indices to KL divergence values.
            - A dictionary mapping layer indices to predictions.
        """
        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)
        tokenizer = self.model.gpt.config.tokenizer

        # Get target logits
        with torch.no_grad():
            model_output: SparsifiedGPTOutput = self.model(input)
        target_logits = model_output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
        target_predictions = get_predictions(tokenizer, target_logits)
        print(f"Target predictions: {target_predictions}")

        # Start with an empty set
        circuit_nodes = frozenset()

        # Add nodes to the circuit by working backwards from the last layer
        for layer_idx in reversed(range(self.num_layers)):
            downstream_nodes = frozenset([node for node in circuit_nodes if node.layer_idx == layer_idx + 1])
            layer_nodes = self.find_layer_nodes(model_output, downstream_nodes, target_token_idx, layer_idx, threshold)
            circuit_nodes = circuit_nodes | layer_nodes

        # Make circuit look more like a tree
        selected_nodes = set(circuit_nodes)
        discarded_nodes = set({})
        for node in sorted(circuit_nodes):
            if node.layer_idx > 0:
                token_idx = node.token_idx
                upstream_idx = node.layer_idx - 1
                if not any(n for n in selected_nodes if n.layer_idx == upstream_idx and n.token_idx == token_idx):
                    selected_nodes.remove(node)
                    discarded_nodes.add(node)
        print(f"\nDiscarded the following nodes: {discarded_nodes}")

        # Calculate final KLD for each layer
        print("\nFinal metrics:")
        klds: dict[int, float] = {}
        predictions: dict[int, dict] = {}
        target_logits = model_output.logits.squeeze(0)[target_token_idx]
        circuit = Circuit(nodes=frozenset(selected_nodes))
        for layer_idx in range(self.num_layers):
            circuit_results = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit],
                model_output.feature_magnitudes[layer_idx].squeeze(0),  # Shape: (T, F)
                self.num_samples * 4,  # Increase samples for final calculation
            )[circuit]
            klds[layer_idx] = circuit_results.kl_divergence
            predictions[layer_idx] = circuit_results.predictions
            print(f"Layer {layer_idx} - KLD: {klds[layer_idx]:.4f} - Predictions: {predictions[layer_idx]}")

        # Return selected nodes, KLDs, and predictions
        return frozenset(selected_nodes), klds, predictions

    def find_layer_nodes(
        self,
        model_output: SparsifiedGPTOutput,
        downstream_nodes: frozenset[Node],
        target_token_idx: int,
        layer_idx: int,
        threshold: float,
    ) -> frozenset[Node]:
        """
        Search for circuit nodes in the selected layer.

        :param model_output: The sparsified model output.
        :param downstream_nodes: The downstream nodes in the circuit.
        :param target_token_idx: The target token index.
        :param layer_idx: The layer index to search in.
        :param threshold: The KL diverence threshold for node extraction.
        """
        # Get output for layer
        feature_magnitudes = model_output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)
        target_logits = model_output.logits.squeeze(0)[target_token_idx]  # Shape: (V)

        # If searching for nodes in the last layer, set start_token_idx to target_token_idx.
        start_token_idx = target_token_idx if layer_idx == self.num_layers - 1 else 0

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
        print(f"\nSearching for tokens in layer {layer_idx} with important information...")
        circuit_candidates = self.select_tokens(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            layer_nodes,
            downstream_nodes,
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
        selected_nodes = set({})
        previous_klds = []
        for node, kld in reversed(ranked_nodes):
            # Stop if KLD is greater than average of previous 3 values
            if len(previous_klds) >= 3 and kld > sum(previous_klds[-3:]) / 3:
                break
            # Add node to layer
            selected_nodes.add(node)
            # Stop if KLD is below search threshold
            if kld < threshold:
                break
            previous_klds.append(kld)

        print(f"Added the following nodes to layer {layer_idx}: {selected_nodes}")
        return frozenset(selected_nodes)

    def rank_nodes(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
    ) -> list[tuple[Node, float]]:
        """
        Rank the nodes in a layer from least to most important.
        """
        # Nodes ordered by when they were removed from the circuit
        ranked_nodes: list[tuple[Node, float]] = []

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
                ranked_nodes.append((node, circuit_analysis.kl_divergence))

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

    def select_tokens(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
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

        # Starting search states
        selected_nodes: frozenset[Node] = frozenset()  # Starting nodes share token indices with downstream nodes
        for downstream_idx in {node.token_idx for node in downstream_nodes}:
            selected_nodes = frozenset(selected_nodes | nodes_by_token_idx[downstream_idx])
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
