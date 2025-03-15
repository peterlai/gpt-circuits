import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfiguration:
    """
    Configuration to use for circuit search.
    """

    k_nearest: int | None = 256  # Number of nearest neighbors to use for clustering
    num_edge_samples: int = 64  # Number of resampling values to use for selecting edges
    num_node_samples: int = 64  # Number of resampling values to use for selecting nodes
    max_positional_coefficient: float = 2.0  # How important is the position of a feature
    max_token_positions: int = 16  # Maximum number of token positions to consider
    rolling_window: int = 7  # Number of previous KLD values to consider for stopping search early


@dataclass(frozen=True)
class Node:
    """
    Represents a feature at a specific location.
    """

    layer_idx: int
    token_idx: int
    feature_idx: int

    def as_tuple(self) -> tuple[int, int, int]:
        return self.layer_idx, self.token_idx, self.feature_idx

    def __repr__(self) -> str:
        return f"({self.layer_idx},{self.token_idx},{self.feature_idx})"

    def __lt__(self, other: "Node") -> bool:
        return self.as_tuple() < other.as_tuple()


@dataclass(frozen=True)
class Edge:
    """
    Represents a connection between two features.
    """

    upstream: Node
    downstream: Node

    def __repr__(self) -> str:
        return f"{self.upstream} -> {self.downstream}"

    def as_tuple(self) -> tuple[int, ...]:
        return self.upstream.as_tuple() + self.downstream.as_tuple()

    def __lt__(self, other: "Edge") -> bool:
        return self.as_tuple() < other.as_tuple()


@dataclass(frozen=True)
class EdgeGroup:
    """
    Represents a group of edges from a downstream token index to an upstream token index.
    """

    upstream_layer_idx: int
    upstream_token_idx: int
    downstream_token_idx: int

    @property
    def downstream_layer_idx(self) -> int:
        return self.upstream_layer_idx + 1

    def __repr__(self) -> str:
        return f"({self.upstream_layer_idx}, {self.upstream_token_idx}) -> ({self.downstream_layer_idx}, {self.downstream_token_idx})"

    def as_tuple(self) -> tuple[int, int, int]:
        return self.upstream_layer_idx, self.upstream_token_idx, self.downstream_token_idx

    def __lt__(self, other: "EdgeGroup") -> bool:
        return self.as_tuple() < other.as_tuple()


@dataclass(frozen=True)
class Circuit:
    """
    Represents a set of nodes and edges.
    """

    nodes: frozenset[Node]
    edges: frozenset[Edge] = frozenset()

    def __repr__(self) -> str:
        return f"Nodes: {sorted(self.nodes)}, Edges: {sorted(self.edges)}"


def json_prettyprint(obj) -> str:
    """
    Return a serialized dictionary as pretty-printed JSON.
    Lists of numbers or strings are formatted using one line.
    """
    serialized_data = json.dumps(obj, indent=2)

    # Regex pattern to remove new lines between "[" and "]"
    pattern = re.compile(r"\[\s*([^{}[\]]{1,10000})\s*\]", re.DOTALL)
    serialized_data = pattern.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", serialized_data)
    return serialized_data
