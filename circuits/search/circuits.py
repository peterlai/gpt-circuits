from dataclasses import dataclass

import torch

from circuits import Circuit, Edge, EdgeGroup, Node, SearchConfiguration
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import ResampleAblator
from circuits.search.divergence import analyze_circuit_divergence
from circuits.search.edges import EdgeSearch
from circuits.search.nodes import NodeSearch
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


@dataclass
class CircuitResult:
    """
    Result of a circuit search.
    """

    circuit: Circuit  # Circuit found
    edge_importances: dict[Edge, float]  # edge -> importance
    token_importances: dict[EdgeGroup, float]  # token-to-token -> importance
    node_importances: dict[Node, float]  # node -> KLD ceiling without node
    node_ranks: dict[Node, int]  # node -> layer rank (1 is the most important)
    positional_coefficients: dict[int, float]  # layer -> positional coefficient
    klds: dict[int, float]  # layer -> KLD
    predictions: dict[int, dict]  # layer -> {token: probability}


class CircuitSearch:
    """
    Search for a circuit in a sparsified model.
    """

    def __init__(
        self,
        model: SparsifiedGPT,
        model_profile: ModelProfile,
        model_cache: ModelCache,
        config: SearchConfiguration,
    ):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param model_profile: The model profile to use for circuit extraction.
        :param model_cache: The model cache to use for circuit extraction.
        :param config: The configuration to use for circuit extraction.
        """
        assert 0 < config.num_edge_samples <= (config.k_nearest or int(1e10))
        assert 0 < config.num_node_samples <= (config.k_nearest or int(1e10))

        self.model = model
        self.model_profile = model_profile
        self.model_cache = model_cache
        self.config = config

    def search(
        self,
        tokens: list[int],
        target_token_idx: int,
        skip_edges: bool = False,
    ) -> CircuitResult:
        """
        Search for a circuit in the model.

        :param tokens: The input tokens to use for circuit extraction.
        :param target_token_idx: The index of the target token to use for circuit extraction.
        :param skip_edges: Whether to skip edge importance calculation.
        """
        # Add nodes to the circuit
        ranked_nodes = frozenset()
        for layer_idx in range(self.num_layers):
            upstream_nodes = frozenset([rn.node for rn in ranked_nodes if rn.node.layer_idx == layer_idx - 1])
            node_search = NodeSearch(self.model, self.create_ablator(layer_idx), self.config)
            layer_nodes = node_search.search(tokens, upstream_nodes, layer_idx, target_token_idx)
            ranked_nodes = ranked_nodes | layer_nodes

        # Iterate through each pairs of consecutive layers to calculate edge importance
        edge_importances: dict[Edge, float] = {}
        token_importances: dict[EdgeGroup, float] = {}
        for upstream_layer_idx in range(self.num_layers - 1):
            upstream_ablator = self.create_ablator(upstream_layer_idx)
            edge_search = EdgeSearch(self.model, self.model_profile, upstream_ablator, self.config.num_edge_samples)
            upstream_nodes = frozenset(rn.node for rn in ranked_nodes if rn.node.layer_idx == upstream_layer_idx)
            downstream_nodes = frozenset(rn.node for rn in ranked_nodes if rn.node.layer_idx == upstream_layer_idx + 1)
            if skip_edges:
                search_result = edge_search.get_placeholders(upstream_nodes, downstream_nodes)
            else:
                search_result = edge_search.search(tokens, upstream_nodes, downstream_nodes, target_token_idx)
            edge_importances.update(search_result.edge_importance)
            token_importances.update(search_result.token_importance)

        # Store positional coefficients
        positional_coefficients = {idx: self.get_positional_coefficient(idx) for idx in range(self.num_layers)}

        # Return circuit
        circuit_nodes = frozenset(rn.node for rn in ranked_nodes)
        circuit_edges = frozenset(edge for edge in edge_importances if edge in circuit_nodes)
        circuit = Circuit(nodes=circuit_nodes, edges=circuit_edges)
        node_importances = {rn.node: rn.kld for rn in ranked_nodes}
        node_ranks = {rn.node: rn.rank for rn in ranked_nodes}
        klds, predictions = self.calculate_klds(circuit, tokens, target_token_idx)

        # Return circuit result
        return CircuitResult(
            circuit,
            edge_importances,
            token_importances,
            node_importances,
            node_ranks,
            positional_coefficients,
            klds,
            predictions,
        )

    def calculate_klds(
        self,
        circuit: Circuit,
        tokens: list[int],
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
        for layer_idx in range(self.num_layers):
            circuit_results = analyze_circuit_divergence(
                self.model,
                self.create_ablator(layer_idx),
                layer_idx,
                target_token_idx,
                target_logits,
                [circuit],
                model_output.feature_magnitudes[layer_idx].squeeze(0),  # Shape: (T, F)
                # Increase samples for final calculation
                min(self.config.num_node_samples * 4, self.config.k_nearest or int(1e10)),
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
        positional_coefficient = self.get_positional_coefficient(layer_idx)

        return ResampleAblator(
            self.model_profile,
            self.model_cache,
            k_nearest=self.config.k_nearest,
            positional_coefficient=positional_coefficient,
        )

    def get_positional_coefficient(self, layer_idx: int) -> float:
        """
        Get the positional coefficient for a given layer.
        """
        # Use smaller positional coefficient for downstream layers ending at 0
        return self.config.max_positional_coefficient * (1.0 - layer_idx / (self.num_layers - 1))

    @property
    def num_layers(self) -> int:
        """
        Get the number of SAE layers in the model.
        """
        return self.model.gpt.config.n_layer + 1  # Add 1 for the embedding layer
