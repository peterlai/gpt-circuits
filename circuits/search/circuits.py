from dataclasses import dataclass

import torch

from circuits import Circuit, Edge, Node
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import analyze_divergence
from circuits.search.edges import EdgeSearch
from circuits.search.nodes import NodeSearch
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


@dataclass(frozen=True)
class CircuitResult:
    """
    Result of a circuit search.
    """

    circuit: Circuit  # Circuit found
    klds: dict[int, float]  # layer idx -> KL divergence using the nodes on that layer
    predictions: dict[int, dict]  # layer idx -> predictions from using nodes on that layer
    edge_importance: dict[Edge, float]  # edge -> importance
    token_importance: dict[Node, dict[int, float]]  # node -> upstream token idx -> importance


class CircuitSearch:
    """
    Search for a circuit in a sparsified model.
    """

    def __init__(
        self,
        model: SparsifiedGPT,
        model_profile: ModelProfile,
        model_cache: ModelCache,
        num_samples: int,
        k_nearest: int | None,
        max_positional_coefficient: float,
    ):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param model_profile: The model profile to use for circuit extraction.
        :param model_cache: The model cache to use for circuit extraction.
        :param num_samples: The number of samples to use for ablation.
        :param k_nearest: The number of nearest neighbors to use for clustering.
        :param max_positional_coefficient: The starting coefficient to use for positional distance when clustering.
        """
        self.model = model
        self.model_profile = model_profile
        self.model_cache = model_cache
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.max_positional_coefficient = max_positional_coefficient

    def search(
        self,
        tokens: list[int],
        target_token_idx: int,
        threshold: float,
    ) -> CircuitResult:
        """
        Search for a circuit in the model.
        """
        # Add nodes to the circuit
        circuit_nodes = frozenset()
        for layer_idx in range(self.num_layers):
            upstream_nodes = frozenset([node for node in circuit_nodes if node.layer_idx == layer_idx - 1])
            node_search = NodeSearch(self.model, self.create_ablator(layer_idx), self.num_samples)
            layer_nodes = node_search.search(tokens, upstream_nodes, layer_idx, target_token_idx, threshold)
            circuit_nodes = circuit_nodes | layer_nodes

        # Make circuit look more like a tree
        # circuit_nodes = self.prune_tree(circuit_nodes)

        # Iterate through each pairs of consecutive layers to calculate edge importance
        edge_importance: dict[Edge, float] = {}
        token_importance: dict[Node, dict[int, float]] = {}
        for upstream_layer_idx in range(self.num_layers - 1):
            upstream_ablator = self.create_ablator(upstream_layer_idx)
            edge_search = EdgeSearch(self.model, self.model_profile, upstream_ablator, self.num_samples)
            upstream_nodes = frozenset(node for node in circuit_nodes if node.layer_idx == upstream_layer_idx)
            downstream_nodes = frozenset(node for node in circuit_nodes if node.layer_idx == upstream_layer_idx + 1)
            search_result = edge_search.search(tokens, upstream_nodes, downstream_nodes, target_token_idx)
            edge_importance.update(search_result.edge_importance)
            token_importance.update(search_result.token_importance)

        # Return circuit
        circuit = Circuit(nodes=circuit_nodes)
        klds, predictions = self.calculate_klds(tokens, circuit_nodes, target_token_idx)
        return CircuitResult(circuit, klds, predictions, edge_importance, token_importance)

    def prune_tree(self, circuit_nodes: frozenset[Node]) -> frozenset[Node]:
        """
        Remove downstream nodes that do not have any upstream nodes directly above them.
        """
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
        return frozenset(selected_nodes)

    def calculate_klds(
        self,
        tokens: list[int],
        circuit_nodes: frozenset[Node],
        target_token_idx: int,
    ) -> tuple[dict[int, float], dict[int, dict]]:
        """
        Calculate the KL divergence for each layer using the nodes on that layer.

        :return: A tuple containing two dictionaries:
            - klds: A dictionary mapping layer indices to KL divergence values.
            - predictions: A dictionary mapping layer indices to predictions.
        """
        # Convert tokens to tensor
        input: torch.Tensor = torch.tensor(tokens, device=self.model.config.device).unsqueeze(0)  # Shape: (1, T)

        # Get target logits
        with torch.no_grad():
            model_output: SparsifiedGPTOutput = self.model(input)

        # Calculate final KLD for each layer
        print("\nKLD metrics:")
        klds: dict[int, float] = {}
        predictions: dict[int, dict] = {}
        target_logits = model_output.logits.squeeze(0)[target_token_idx]
        circuit = Circuit(nodes=circuit_nodes)
        for layer_idx in range(self.num_layers):
            circuit_results = analyze_divergence(
                self.model,
                self.create_ablator(layer_idx),
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

        return klds, predictions

    def create_ablator(self, layer_idx: int) -> ResampleAblator:
        """
        Create an ablator for the model.

        :param layer_idx: The layer index from which feature magnitudes will be ablated.
        """
        # Use lower positional coefficient for downstream layers
        positional_coefficient = self.max_positional_coefficient * (1.0 - layer_idx / (self.num_layers - 1))

        return ResampleAblator(
            self.model_profile,
            self.model_cache,
            k_nearest=self.k_nearest,
            positional_coefficient=positional_coefficient,
        )

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1  # Add 1 for the embedding layer
