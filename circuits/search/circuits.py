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


@dataclass
class CircuitResult:
    """
    Result of a circuit search.
    """

    circuit: Circuit  # Circuit found
    edge_importance: dict[Edge, float]  # edge -> importance
    token_importance: dict[Node, dict[int, float]]  # node -> upstream token idx -> importance
    node_importance: dict[Node, float]  # node -> KLD ceiling without node


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
        ranked_nodes = frozenset()
        for layer_idx in range(self.num_layers):
            upstream_nodes = frozenset([rn.node for rn in ranked_nodes if rn.node.layer_idx == layer_idx - 1])
            node_search = NodeSearch(self.model, self.create_ablator(layer_idx), self.num_samples)
            layer_nodes = node_search.search(tokens, upstream_nodes, layer_idx, target_token_idx, threshold)
            ranked_nodes = ranked_nodes | layer_nodes

        # Iterate through each pairs of consecutive layers to calculate edge importance
        edge_importance: dict[Edge, float] = {}
        token_importance: dict[Node, dict[int, float]] = {}
        for upstream_layer_idx in range(self.num_layers - 1):
            upstream_ablator = self.create_ablator(upstream_layer_idx)
            edge_search = EdgeSearch(self.model, self.model_profile, upstream_ablator, self.num_samples)
            upstream_nodes = frozenset(rn.node for rn in ranked_nodes if rn.node.layer_idx == upstream_layer_idx)
            downstream_nodes = frozenset(rn.node for rn in ranked_nodes if rn.node.layer_idx == upstream_layer_idx + 1)
            search_result = edge_search.search(tokens, upstream_nodes, downstream_nodes, target_token_idx)
            edge_importance.update(search_result.edge_importance)
            token_importance.update(search_result.token_importance)

        # Return circuit
        circuit_nodes = frozenset(rn.node for rn in ranked_nodes)
        circuit_edges = frozenset(edge for edge in edge_importance if edge in circuit_nodes)
        circuit = Circuit(nodes=circuit_nodes, edges=circuit_edges)
        node_importance = {rn.node: rn.kld for rn in ranked_nodes}
        return CircuitResult(circuit, edge_importance, token_importance, node_importance)

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
