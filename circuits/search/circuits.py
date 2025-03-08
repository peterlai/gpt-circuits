from dataclasses import dataclass

from circuits import Circuit, Edge, Node
from circuits.features.profiles import ModelProfile
from circuits.search.ablation import Ablator
from circuits.search.edges import EdgeSearch
from circuits.search.nodes import NodeSearch
from models.sparsified import SparsifiedGPT


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

    def __init__(self, model: SparsifiedGPT, model_profile: ModelProfile, ablator: Ablator, num_samples: int):
        """
        :param model: The sparsified model to use for circuit extraction.
        :param model_profile: The model profile to use for circuit extraction.
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
        threshold: float,
    ) -> CircuitResult:
        """
        Search for a circuit in the model.
        """
        # Get circuit nodes
        node_search = NodeSearch(self.model, self.ablator, self.num_samples)
        circuit_nodes, klds, predictions = node_search.search(tokens, target_token_idx, threshold)

        # Find circuit edges
        edge_search = EdgeSearch(self.model, self.model_profile, self.ablator, self.num_samples)
        edge_importance, token_importance = edge_search.search(tokens, target_token_idx, circuit_nodes)

        # Return circuit
        circuit = Circuit(nodes=circuit_nodes)
        return CircuitResult(circuit, klds, predictions, edge_importance, token_importance)
