import json
import math
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional

import torch
from scipy import sparse
from tqdm import tqdm

from data.dataloaders import DatasetShard
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput


class ModelCache:
    """
    Contains cached feature magnitudes for a model.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Cached feature magnitudes for every layer of a model.
        """
        self.split = ""
        self.shard_idx = 0
        self.block_size = 0
        self.num_layers = 0
        self.layers: dict[int, LayerCache] = {}

        # Load from checkpoint if provided
        if checkpoint_dir:
            if not (checkpoint_dir / self.filename).exists():
                raise FileNotFoundError("Metrics don't exist. Please run `compute_metrics`.")

            # Load metadata
            with open(checkpoint_dir / self.filename, "r") as f:
                meta = json.load(f)
                self.split = meta["split"]
                self.shard_idx = meta["shard_idx"]
                self.block_size = meta["block_size"]
                self.num_layers = meta["num_layers"]

            # Load layers
            for layer_idx in range(self.num_layers):
                layer_cache = LayerCache(layer_idx, self.block_size)
                layer_cache.load(checkpoint_dir)
                self.layers[layer_idx] = layer_cache

    def __getitem__(self, layer_idx: int) -> "LayerCache":
        """
        Get the layer cache for a given layer index.
        """
        return self.layers[layer_idx]

    @torch.no_grad()
    def compute(self, model: SparsifiedGPT, shard: DatasetShard, batch_size: int):
        """
        Compute feature magnitudes to cache for a model.
        """
        self.split = shard.split
        self.shard_idx = shard.shard_idx
        self.block_size = model.config.block_size
        self.num_layers = len(model.config.n_features)

        # Set model to evaluation mode
        model.eval()

        # Batched model outputs
        batch_feature_magnitudes: dict[int, list[sparse.csr_matrix]] = defaultdict(list)

        # Walk through shard.tokens in batches
        tokens_per_batch = batch_size * self.block_size
        num_blocks = len(shard.tokens) // self.block_size
        num_batches = math.ceil(num_blocks / batch_size)
        starting_idxs = range(0, num_blocks * self.block_size, tokens_per_batch)
        for i in tqdm(starting_idxs, total=num_batches, desc="Computing feature magnitudes"):

            # Get batch of tokens
            ending_idx = min(shard.tokens.size(-1), i + tokens_per_batch)
            ending_idx -= ending_idx % self.block_size
            tokens = shard.tokens[i:ending_idx].to(model.config.device)
            tokens = tokens.view(-1, self.block_size)

            # Get feature magnitudes
            output: SparsifiedGPTOutput = model(tokens)

            for layer_idx, feature_magnitudes in output.feature_magnitudes.items():
                # Remove batch dimension
                flattened_feature_magnitudes = feature_magnitudes.view(-1, feature_magnitudes.size(-1))
                # Append sparse matrix
                sparse_matrix = sparse.csr_matrix(flattened_feature_magnitudes.cpu().numpy())
                batch_feature_magnitudes[layer_idx].append(sparse_matrix)

        # Cache layers
        for layer_idx, stackable_magnitudes in batch_feature_magnitudes.items():
            # Stack token magnitudes
            coo_feature_magnitudes: sparse.coo_matrix = sparse.vstack(stackable_magnitudes, format="coo")  # type: ignore
            layer_cache = LayerCache(layer_idx, self.block_size)
            layer_cache.update(coo_feature_magnitudes)
            self.layers[layer_idx] = layer_cache

    def save(self, checkpoint_dir: Path):
        """
        Save feature magnitudes to disk.
        """
        # Save metadata
        meta = {
            "split": self.split,
            "shard_idx": self.shard_idx,
            "block_size": self.block_size,
            "num_layers": self.num_layers,
        }
        with open(checkpoint_dir / self.filename, "w") as json_file:
            json.dump(meta, json_file)

        # Save layers
        for layer_cache in tqdm(self.layers.values(), desc="Caching feature magnitudes"):
            layer_cache.save(checkpoint_dir)

    @property
    def filename(self) -> str:
        """
        Return the output file path.
        """
        return "metrics.magnitudes.json"

    def get_shard(self, data_dir: Path) -> DatasetShard:
        """
        Return the shard that was used to compute the feature magnitudes.
        """
        return DatasetShard(
            dir_path=data_dir,
            split=self.split,
            shard_idx=self.shard_idx,
        )


class LayerCache:
    """
    Contains cached feature magnitudes for a layer.
    """

    def __init__(self, layer_idx: int, block_size):
        """
        Create empty layer magnitudes cache.
        """
        self.layer_idx = layer_idx
        self.block_size = block_size  # Block size used to compute feature magnitudes
        self.magnitudes: sparse.coo_matrix = sparse.coo_matrix((0, 0))  # Shape: (num_tokens, num_features)

    @cached_property
    def csc_matrix(self) -> sparse.csc_matrix:
        """
        Return the feature magnitudes in CSC format for faster column slicing.
        """
        return self.magnitudes.tocsc()  # type: ignore

    @cached_property
    def csr_matrix(self) -> sparse.csr_matrix:
        """
        Return the feature magnitudes in CSR format for faster row slicing.
        """
        return self.magnitudes.tocsr()  # type: ignore

    def update(self, coo_feature_magnitudes: sparse.coo_matrix):
        """
        Update feature magnitudes.
        """
        self.magnitudes = coo_feature_magnitudes

    def save(self, checkpoint_dir: Path):
        """
        Save feature magnitudes to disk.
        """
        sparse.save_npz(checkpoint_dir / self.filename, self.magnitudes)

    def load(self, checkpoint_dir: Path):
        """
        Load feature magnitudes from disk.
        """
        self.magnitudes = sparse.load_npz(checkpoint_dir / self.filename)

    @property
    def filename(self) -> str:
        """
        Return the output file path for NPZ data.
        """
        return f"metrics.magnitudes.{self.layer_idx}.npz"
