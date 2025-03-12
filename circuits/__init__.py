import json
import re
from dataclasses import dataclass


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

#########################################################################

@dataclass(frozen=True)
class TokenlessNode:
    """
    Represents a feature at a specific location without token information.
    """
    layer_idx: int
    feature_idx: int

    def as_tuple(self) -> tuple[int, int]:
        return self.layer_idx, self.feature_idx

    def __repr__(self) -> str:
        return f"({self.layer_idx},{self.feature_idx})"

    def __lt__(self, other: "TokenlessNode") -> bool:
        return self.as_tuple() < other.as_tuple()

@dataclass(frozen=True)
class TokenlessEdge:
    """
    Represents a connection between two features without token information.
    """
    upstream: TokenlessNode
    downstream: TokenlessNode

    def __repr__(self) -> str:
        return f"{self.upstream} -> {self.downstream}"

    def as_tuple(self) -> tuple[int, ...]:
        return self.upstream.as_tuple() + self.downstream.as_tuple()

    def __lt__(self, other: "TokenlessEdge") -> bool:
        return self.as_tuple() < other.as_tuple()
    
@dataclass(frozen=True)
class TokenlessCircuit:
    """
    Represents a set of nodes and edges without token information.
    """
    nodes: frozenset[TokenlessNode]
    edges: frozenset[TokenlessEdge] = frozenset()

    def __repr__(self) -> str:
        return f"Nodes: {sorted(self.nodes)}, Edges: {sorted(self.edges)}"
