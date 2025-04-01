import torch
import torch.nn.functional as Fn
import random
import numpy as np
import glob
import os
from safetensors.torch import load_file
import json
import re

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

    edges = frozenset([
        TokenlessEdge(
            upstream=TokenlessNode(layer_idx=upstream_layer_idx, feature_idx=f_up),
            downstream=TokenlessNode(layer_idx=upstream_layer_idx+1, feature_idx=f_down)
        )
        for f_up, f_down in edge_arr
    ])
    
    return edges


def get_attribution_rankings(attribution_tensor):

    flattened = attribution_tensor.flatten()
    
    # Get the indices that would sort the array in descending order (top N only)
    sorted_indices = torch.argsort(flattened, descending=True)
    
    # Store the top indices and their values in a structured way
    ranked_indices = []
    attribution_values = []
    
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i].item()
        # Convert flat index to 2D coordinates
        row, col = idx // attribution_tensor.shape[1], idx % attribution_tensor.shape[1]
        ranked_indices.append((row, col))
        attribution_values.append(attribution_tensor[row, col].item())
    
    return ranked_indices, attribution_values

def load_experiments_and_extract_data(exp_output, data_dir, sae_variant, edge_sort, layers):

    """
    Extract data from experiments based on the specified parameters.

    :param exp_output: Experimental data to load ('feature_magnitudes', 'logits' or 'kl_divergence').
    :param project_root:
    :param run_dir:
    :param sae_variant:
    :param edge_sort:
    :param layers:

    """
    # Set pattern template for experiment IDs
    pattern_template = "magnitudes_" + f"{sae_variant}_" + f"{edge_sort}_" +  "{layer}_"

    # Get all safetensor files in the directory
    safetensor_files = glob.glob(str(data_dir / "*.safetensors"))

    # Dictionary to store all loaded experiments
    experiments = {}

    # Load each experiment
    for file_path in safetensor_files:
        # Load tensors
        tensors = load_file(file_path)
        
        # Load metadata if it exists
        metadata_path = file_path + ".metadata.json"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Extract experiment ID from filename
        filename = os.path.basename(file_path)
        experiment_id = filename.split("_")[:-1]  # Remove timestamp part
        experiment_id = "_".join(experiment_id)
        
        # Store in dictionary
        experiments[experiment_id] = {
            "tensors": tensors,
            "metadata": metadata,
            "file_path": file_path
        }

    print(f"Loaded {len(experiments)} experiments from {data_dir}")

    # Example of accessing data from the first experiment
    if experiments:
        first_exp_id = list(experiments.keys())[0]
        print(f"\nExample experiment ID: {first_exp_id}")
        print(f"Available tensor keys: {list(experiments[first_exp_id]['tensors'].keys())}")
        
        # If feature_magnitudes exists, print its shape
        if 'feature_magnitudes' in experiments[first_exp_id]['tensors']:
            feat_mag = experiments[first_exp_id]['tensors']['feature_magnitudes']
            print(f"Feature magnitudes shape: {feat_mag.shape}")
        
        # If logits exists, print its shape
        if 'logits' in experiments[first_exp_id]['tensors']:
            logits = experiments[first_exp_id]['tensors']['logits']
            print(f"Logits shape: {logits.shape}")

        # If kl_divergence exists, print its shape
        if 'kl_divergence' in experiments[first_exp_id]['tensors']:
            kl_div = experiments[first_exp_id]['tensors']['kl_divergence']
            print(f"KL divergence shape: {kl_div.shape}")

    # Process each layer
    layer_data_values = []

    for layer in layers:
        # Extract all experiments with the specified pattern
        random_level_exps = {}
        pattern = re.compile(pattern_template.format(layer=layer) + r'(\d+)')

        for exp_id, exp_data in experiments.items():
            if f'{edge_sort}' in exp_id and exp_id.startswith(pattern_template.format(layer=layer)):
                match = pattern.search(exp_id)
                if match:
                    num_features = int(match.group(1))
                    random_level_exps[num_features] = exp_data

        # Sort experiments by number of features
        sorted_exps = sorted(random_level_exps.items())

        # Extract feature counts and exp_output values
        feature_counts = []
        random_data_values = []

        for num_features, exp_data in sorted_exps:
            feature_counts.append(num_features)

            # Get exp_output values
            data = exp_data['tensors'][exp_output]
            random_data_values.append(data)

        # Store results for this layer
        layer_data_values.append(random_data_values)
    
    return layer_data_values, feature_counts


def bootstrap_ci(data, n_bootstrap=1000, statistic=np.mean, confidence=0.95):
    """
    Calculate bootstrap confidence interval for the given statistic.
    
    Parameters:
    -----------
    data : array-like
        The data to bootstrap
    n_bootstrap : int
        Number of bootstrap samples to generate
    statistic : function
        The statistic to compute (e.g., np.mean, np.median)
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    tuple
        Lower and upper bounds of the confidence interval
    """
    bootstrap_stats = []
    
    # Create bootstrap samples and compute statistics
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=len(data), replace=True)
        # Calculate statistic
        bootstrap_stats.append(statistic(resampled))
    
    # Sort the bootstrap statistics
    bootstrap_stats.sort()
    
    # Calculate the confidence interval
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap)
    
    # Return the confidence interval
    return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]