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
        self.feature_num_samples = num_samples
        self.token_num_samples = num_samples * 2  # Use more samples for token search to reduce variance

    def search(
        self,
        tokens: Sequence[int],
        layer_idx: int,
        start_token_idx: int,
        target_token_idx: int,
        threshold: float,
    ) -> frozenset[Node]:
        """
        Search for circuit nodes in a sparsified model.

        :param tokens: The token inputs.
        :param layer_idx: The layer index to search in.
        :param start_token_idx: The token index to start search from.
        :param target_token_idx: The target token index.
        :param threshold: The KL diverence threshold for node extraction.
        """
        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)
        tokenizer = self.model.gpt.config.tokenizer

        # Get target logits
        with torch.no_grad():
            output: SparsifiedGPTOutput = self.model(input)
        target_logits = output.logits.squeeze(0)[target_token_idx]  # Shape: (V)
        target_predictions = get_predictions(tokenizer, target_logits)
        print(f"Target predictions: {target_predictions}")

        # Get baseline KL divergence
        x_reconstructed = self.model.saes[str(layer_idx)].decode(output.feature_magnitudes[layer_idx])  # type: ignore
        predicted_logits = self.model.gpt.forward_with_patched_activations(x_reconstructed, layer_idx=layer_idx)
        predicted_logits = predicted_logits[0, target_token_idx, :]  # Shape: (V)
        baseline_predictions = get_predictions(tokenizer, predicted_logits)
        baseline_kl_div: float = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(target_logits, dim=-1),
            torch.nn.functional.softmax(predicted_logits, dim=-1),
            reduction="sum",
        ).item()
        print(f"Baseline predictions: {baseline_predictions}")
        print(f"Baseline KL divergence: {baseline_kl_div:.4f}\n")

        # Get output for layer
        feature_magnitudes = output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)

        # If searching for nodes in the last layer, set start_token_idx to target_token_idx.
        # The last layer is special because its output for the target token is directly used to produce logits.
        if layer_idx == self.model.gpt.config.n_layer:
            start_token_idx = target_token_idx

        # Get non-zero features where token index is in [start_token_idx...target_token_idx]
        all_nodes: set[Node] = set({})
        non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
        assert start_token_idx <= target_token_idx
        for t, f in zip(*non_zero_indices):
            if t >= start_token_idx and t <= target_token_idx:
                all_nodes.add(Node(layer_idx, t.item(), f.item()))

        ### Part 1: Start by searching for important tokens
        print("Starting search for important tokens...")
        circuit_nodes = self.select_tokens(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            all_nodes=frozenset(all_nodes),
            threshold=threshold / 2,  # Use a lower threshold for coarse search
        )

        ### Part 2: Search for important features
        print("\nStarting search for important features...")
        circuit_nodes = self.prune_features(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            circuit_nodes=circuit_nodes,
            threshold=threshold,
        )
        return circuit_nodes

    def select_tokens(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        all_nodes: frozenset[Node],
        threshold: float,
    ) -> frozenset[Node]:
        """
        Select tokens to be represented in the circuit.
        """
        # Group features by token index
        nodes_by_token_idx: dict[int, set[Node]] = {}
        for token_idx in range(target_token_idx + 1):
            nodes_by_token_idx[token_idx] = set({node for node in all_nodes if node.token_idx == token_idx})

        # Starting search states
        circuit_nodes: frozenset[Node] = frozenset()  # Start with empty circuit
        new_nodes: set[Node] = set({})
        previous_kl_divs: list[float] = []

        # Start search
        for _ in range(target_token_idx + 1):
            # Update circuit
            circuit_nodes = frozenset(circuit_nodes | new_nodes)

            # Compute KL divergence
            circuit = Circuit(nodes=circuit_nodes)
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit],
                feature_magnitudes,
                num_samples=self.token_num_samples,
            )[circuit]

            # Print results
            print(
                f"Tokens: {len(set(f.token_idx for f in circuit.nodes))}/{target_token_idx + 1} - "
                f"KL Div: {circuit_analysis.kl_divergence:.4f} - "
                f"Added: {set([n.token_idx for n in new_nodes]) or 'None'} - "
                f"Predictions: {circuit_analysis.predictions}"
            )

            # Stop early if KL divergence has not improved after 8 iterations
            if len(previous_kl_divs) >= 8 and circuit_analysis.kl_divergence > previous_kl_divs[-8]:
                print("Stopping search - can't improve KL divergence.")
                break

            # Store KL divergence
            previous_kl_divs.append(circuit_analysis.kl_divergence)

            # If below threshold, stop search
            if circuit_analysis.kl_divergence < threshold:
                print("Stopping search - Reached target KL divergence.")
                break

            # Sort tokens by KL divergence (descending)
            estimated_token_importance = self.estimate_token_importance(
                layer_idx,
                target_token_idx,
                target_logits,
                feature_magnitudes,
                circuit_nodes=circuit_nodes,
                remaining_nodes=all_nodes - circuit_nodes,
            )
            most_important_token_idx = min(estimated_token_importance.items(), key=lambda x: x[1])[0]
            new_nodes = nodes_by_token_idx[most_important_token_idx]

        # Print results (grouped by token_idx)
        print(f"\nCircuit has {len(circuit_nodes)} nodes after token search on layer {layer_idx}:")
        for token_idx in range(max([node.token_idx for node in circuit_nodes]) + 1):
            nodes = [node for node in circuit_nodes if node.token_idx == token_idx]
            if len(nodes) > 0:
                print(f"Token {token_idx}: {', '.join([str(node.feature_idx) for node in nodes])}")

        # Return circuit
        return circuit_nodes

    def prune_features(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_nodes: frozenset[Node],
        threshold: float,
    ) -> frozenset[Node]:
        """
        Prune features from the circuit.
        """
        # Starting search states
        initial_nodes = circuit_nodes
        discard_candidates: set[Node] = set({})
        circuit_kl_div: float = float("inf")

        # Start search
        for _ in range(len(circuit_nodes)):
            # Compute KL divergence
            circuit_candidate = Circuit(nodes=frozenset(circuit_nodes - discard_candidates))
            circuit_analysis = analyze_divergence(
                self.model,
                self.ablator,
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit_candidate],
                feature_magnitudes,
                num_samples=self.feature_num_samples,
            )[circuit_candidate]
            circuit_kl_div = circuit_analysis.kl_divergence

            # If below threshold, continue search
            if circuit_kl_div < threshold:
                # Update circuit
                circuit_nodes = circuit_candidate.nodes

                # Find discard candidates
                discard_candidates = self.find_least_important_nodes(
                    layer_idx,
                    target_token_idx,
                    target_logits,
                    feature_magnitudes,
                    circuit_nodes=circuit_nodes,
                    # Remove 3% of nodes on each iteration
                    max_count=int(math.ceil(len(circuit_nodes) * 0.03)),
                )

                # Print results
                print(
                    f"Features: {len(circuit_nodes)}/{len(initial_nodes)} - "
                    f"KL Div: {circuit_kl_div:.4f} - "
                    f"Discarded: {set([(n.token_idx, n.feature_idx) for n in discard_candidates])} - "
                    f"Predictions: {circuit_analysis.predictions}"
                )

            # If above threshold, stop search
            else:
                print("Stopping search - Reached target KL divergence.")
                break

        # Print final results (grouped by token_idx)
        print(f"\nCircuit has {len(circuit_nodes)} nodes after feature search on layer {layer_idx}:")
        for token_idx in range(max([f.token_idx for f in circuit_nodes]) + 1):
            nodes = [f for f in circuit_nodes if f.token_idx == token_idx]
            if len(nodes) > 0:
                print(f"Token {token_idx}: {', '.join([str(f.feature_idx) for f in nodes])}")

        # Return circuit
        return circuit_nodes

    def find_least_important_nodes(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_nodes: frozenset[Node],
        max_count: int,
    ) -> set[Node]:
        """
        Map features to KL divergence.
        """
        # Generate all circuit variations with one node removed
        circuit_variants: dict[Node, Circuit] = {}
        for node in circuit_nodes:
            circuit_variants[node] = Circuit(nodes=frozenset([n for n in circuit_nodes if n != node]))

        # Calculate KL divergence for each variant
        kld_results = analyze_divergence(
            self.model,
            self.ablator,
            layer_idx,
            target_token_idx,
            target_logits,
            [variant for variant in circuit_variants.values()],
            feature_magnitudes,
            self.feature_num_samples,
        )

        # Map nodes to KL divergence
        node_to_kld = {node: kld_results[variant].kl_divergence for node, variant in circuit_variants.items()}

        # Find least important nodes
        sorted_nodes = sorted(node_to_kld.items(), key=lambda x: x[1])  # Sort by KL divergence (ascending)
        least_important_nodes = set([node for node, _ in sorted_nodes[:max_count]])
        return least_important_nodes

    def estimate_token_importance(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        circuit_nodes: frozenset[Node],
        remaining_nodes: frozenset[Node],
    ) -> dict[int, float]:
        """
        Map tokens to KL divergence.
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
            self.token_num_samples,
        )

        # Map token indices to KL divergence
        results = {token_idx: kld_results[variant].kl_divergence for token_idx, variant in circuit_variants.items()}
        return results
