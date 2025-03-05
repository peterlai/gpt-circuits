import math
from typing import Sequence

import torch

from circuits import Circuit, Node
from circuits.search.ablation import Ablator
from circuits.search.divergence import analyze_divergence, get_predictions
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class CircuitSearch:
    """
    Search for a circuit in a sparsified model.
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
        tokens: Sequence[int],
        target_token_idx: int,
        threshold: float,
    ) -> Circuit:
        """
        Search for a circuit in the model.
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

        # Start with an empty circuit
        circuit = Circuit(frozenset(), frozenset())

        # Add nodes to the circuit by working backwards from the last layer
        for layer_idx in reversed(range(self.num_layers)):
            layer_nodes = self.find_nodes(model_output, circuit, target_token_idx, layer_idx, threshold)
            circuit = Circuit(circuit.nodes | layer_nodes, frozenset())

        return circuit

    def find_nodes(self, model_output: SparsifiedGPTOutput, circuit: Circuit, target_token_idx: int, layer_idx: int, threshold: float) -> frozenset[Node]:
        """
        Find the nodes in a particular layer that should be added to a circuit.
        """
        # Which downstream nodes should we consider?
        downstream_nodes = frozenset([node for node in circuit.nodes if node.layer_idx == layer_idx + 1])

        # Get output for layer
        feature_magnitudes = model_output.feature_magnitudes[layer_idx].squeeze(0)  # Shape: (T, F)
        target_logits = model_output.logits.squeeze(0)[target_token_idx]  # Shape: (V)

        # If searching for nodes in the last layer, set start_token_idx to target_token_idx.
        start_token_idx = target_token_idx if layer_idx == self.num_layers - 1 else 0

        # Get non-zero features where token index is in [start_token_idx...target_token_idx]
        all_nodes: set[Node] = set({})
        non_zero_indices = torch.nonzero(feature_magnitudes, as_tuple=True)
        for t, f in zip(*non_zero_indices):
            if t >= start_token_idx and t <= target_token_idx:
                all_nodes.add(Node(layer_idx, t.item(), f.item()))

        # Narrow down nodes to consider
        circuit_candidates = frozenset(all_nodes) # TODO: Implement

        # Rank remaining features by importance
        print("\nRanking features...")
        ranked_nodes = self.rank_features(
            layer_idx,
            target_token_idx,
            target_logits,
            feature_magnitudes,
            circuit_candidates,
            downstream_nodes,
        )

        # Walk backwards through ranked nodes and stop when KL divergence is below threshold.
        # NOTE: KL divergence does not monotonically decrease, which is why we need to walk backwards.
        layer_nodes = set({})
        previous_klds = []
        for node, kld in reversed(ranked_nodes):
            layer_nodes.add(node)
            if kld < threshold:
                break
            # Also stop if KLD is greater than average of previous 8 values
            elif len(previous_klds) >= 8 and kld > sum(previous_klds[-8:]) / 8:
                break
            previous_klds.append(kld)

        return frozenset(layer_nodes)

    def rank_features(
        self,
        layer_idx: int,
        target_token_idx: int,
        target_logits: torch.Tensor,
        feature_magnitudes: torch.Tensor,
        layer_nodes: frozenset[Node],
        downstream_nodes: frozenset[Node],
    ) -> list[tuple[Node, float]]:
        """
        Rank the features in a layer from least to most important.
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
                downstream_nodes=downstream_nodes,
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
        downstream_nodes: frozenset[Node],
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

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1 # Add 1 for the embedding layer