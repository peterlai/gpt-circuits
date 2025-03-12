import torch
import torch.nn.functional as Fn
import random
import numpy as np

from circuits import Circuit, Edge, Node, TokenlessNode, TokenlessEdge

def compute_kl_divergence(logits_p, logits_q):
    """
    Compute KL(P||Q) where P and Q are probability distributions defined by the logits
    
    Parameters:
    -----------
    logits_p : Tensor of shape (vocab_size)
        First set of logits (P distribution)
    logits_q : Tensor of shape (vocab_size)
        Second set of logits (Q distribution)
        
    Returns:
    --------
    float : KL divergence value
    """
    # Convert logits to probabilities
    p = Fn.softmax(logits_p, dim=-1)
    q = Fn.softmax(logits_q, dim=-1)
    
    # KL divergence: sum(p * log(p/q))
    kl_div = Fn.kl_div(q.log(), p, reduction='sum')
    
    return kl_div.item()

def create_random_edges(
    layer_l: int, 
    num_features_l: int, 
    num_features_l_plus_1: int, 
    num_edges: int,
    tokens: list[int]=None,
    seed: int=None
):
    """
    Randomly select a specified number of edges between layer L and layer L+1
    
    Parameters:
    -----------
    layer_l : int
        The index of the upstream layer
    num_features_l : int
        Number of features in layer L
    num_features_l_plus_1 : int
        Number of features in layer L+1
    num_edges : int
        Number of edges to select randomly
    tokens : list[int], optional
        Token indices to include. If None, only uses token idx 0
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    frozenset[Edge] : Randomly selected edges
    """
    assert 0 <= num_edges <= num_features_l*num_features_l_plus_1, "Number of edges must be greater than 0"
    
    if tokens is None:
        tokens = [0]  # Default to just token 0
    
    if seed is not None:
        random.seed(seed)
    
    # Create all possible edges as tuples (for faster random selection)
    all_edge_tuples = [
        (layer_l, t, f_up, layer_l+1, t, f_down)
        for t in tokens
        for f_up in range(num_features_l)
        for f_down in range(num_features_l_plus_1)
    ]
    
    # Calculate total possible edges
    total_possible = len(all_edge_tuples)
    
    # Make sure we don't select more edges than possible
    num_to_select = min(num_edges, total_possible)
    
    # Randomly select edges
    selected_tuples = random.sample(all_edge_tuples, num_to_select)
    
    # Convert back to Edge objects
    edges = frozenset([
        Edge(
            upstream=Node(layer_idx=l_up, token_idx=t_up, feature_idx=f_up),
            downstream=Node(layer_idx=l_down, token_idx=t_down, feature_idx=f_down)
        )
        for l_up, t_up, f_up, l_down, t_down, f_down in selected_tuples
    ])
    
    return edges


def create_sparse_dense_tensor(*dimensions, sparsity=0.2, dtype=torch.float32, device="cpu"):
    """
    Create a regular PyTorch tensor with sparse entries (mostly zeros) of arbitrary dimensions.
    
    :param *dimensions: Variable number of dimension sizes (e.g., 10, 20, 30 for a 3D tensor)
    :param sparsity (float): Sparsity level (0.0 to 1.0), proportion of zeros in the tensor
    :param dtype: Data type of the tensor (default: torch.float32)
    :param device: Device to place tensor on (default: "cpu")
    :return: torch.Tensor: A regular tensor with sparse entries
    """
    # Create a tensor of uniform random values
    random_tensor = torch.rand(*dimensions, dtype=dtype, device=device)
    
    # Create a mask where values > sparsity will be kept (non-zero)
    mask = random_tensor > sparsity
    
    # Generate values for non-zero elements
    values = torch.randn(*dimensions, dtype=dtype, device=device)
    
    # Apply the mask to get a sparse structure
    sparse_dense_tensor = values * mask
    
    return sparse_dense_tensor

def randomly_select_edges(edges: frozenset[Edge], num_edges: int) -> frozenset[Edge]:
    """
    Randomly select num_edges from the given frozen set of edges.
    
    Args:
        edges: The original set of edges
        num_edges: Number of edges to select
        
    Returns:
        A new frozen set with num_edges randomly selected edges
        
    Raises:
        ValueError: If num_edges is greater than the number of edges
    """
    # Convert to list for random sampling
    edge_list = list(edges)
    total_edges = len(edge_list)
    
    if num_edges > total_edges:
        raise ValueError(f"Cannot select {num_edges} edges from a set of {total_edges} edges")
    
    if num_edges == total_edges:
        return edges
    
    if num_edges == 0:
        return frozenset()
    
    # Randomly select num_edges
    selected_edges = random.sample(edge_list, num_edges)
    
    # Return as frozenset
    return frozenset(selected_edges)


def create_full_edge_set(
    upstream_layer_idx: int, 
    num_upstream_features: int, 
    num_downstream_features: int, 
    target_token_idx: int
):


    # Create all possible edges
    ## THINK ABOUT WHY WE NEED ALL POSSIBLE EDGES
    edges = frozenset([
        Edge(
            upstream=Node(layer_idx=upstream_layer_idx, token_idx=t_up, feature_idx=f_up),
            downstream=Node(layer_idx=upstream_layer_idx+1, token_idx=t_down, feature_idx=f_down)
        )
        for t_up in range(target_token_idx+1)
        for t_down in range(target_token_idx+1)
        for f_up in range(num_upstream_features)
        for f_down in range(num_downstream_features)
    ])
    
    return edges


def select_edges_from_array(
    edge_arr: np.ndarray, 
    upstream_layer_idx: int,
    target_token_idx: int
):


    # Create all possible edges
    ## THINK ABOUT WHY WE NEED ALL POSSIBLE EDGES -- make this part more efficient
    edges = frozenset([
        Edge(
            upstream=Node(layer_idx=upstream_layer_idx, token_idx=t_up, feature_idx=f_up),
            downstream=Node(layer_idx=upstream_layer_idx+1, token_idx=t_down, feature_idx=f_down)
        )
        for t_up in range(target_token_idx+1)
        for t_down in range(target_token_idx+1)
        for f_up, f_down in edge_arr
    ])
    
    return edges


def create_tokenless_edges_from_array(
    edge_arr: np.ndarray, 
    upstream_layer_idx: int
):


    # Create all possible edges
    ## THINK ABOUT WHY WE NEED ALL POSSIBLE EDGES -- make this part more efficient
    edges = frozenset([
        TokenlessEdge(
            upstream=TokenlessNode(layer_idx=upstream_layer_idx, feature_idx=f_up),
            downstream=TokenlessNode(layer_idx=upstream_layer_idx+1, feature_idx=f_down)
        )
        for f_up, f_down in edge_arr
    ])

    # edges = frozenset([
    #     Edge(
    #         upstream=Node(layer_idx=upstream_layer_idx, token_idx=t_up, feature_idx=f_up),
    #         downstream=Node(layer_idx=upstream_layer_idx+1, token_idx=target_token_idx, feature_idx=f_down)
    #     )
    #     for t_up in range(target_token_idx+1)
    #     for f_up, f_down in edge_arr
    # ])
    
    return edges