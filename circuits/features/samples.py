import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

from circuits import json_prettyprint
from circuits.features.cache import LayerCache, ModelCache
from data.dataloaders import DatasetShard
from data.tokenizers import Tokenizer


class ModelSampleSet:
    """
    Contains feature samples for a model.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Cached feature profiles for every layer of a model.
        """
        self.layers: dict[int, LayerSampleSet] = {}

        # Load from checkpoint if provided
        if checkpoint_dir is not None:
            for layer_idx in range(self.get_num_layers(checkpoint_dir)):
                layer_profile = LayerSampleSet(layer_idx)
                layer_profile.load(checkpoint_dir)
                self.layers[layer_idx] = layer_profile

    def __getitem__(self, layer_idx: int) -> "LayerSampleSet":
        """
        Get the layer sample set for a given layer index.
        """
        return self.layers[layer_idx]

    def __iter__(self) -> Iterator["LayerSampleSet"]:
        """
        Return an iterator over the layer sample sets.
        """
        return iter(self.layers.values())

    def __len__(self) -> int:
        """
        Return the number of layer sample sets.
        """
        return len(self.layers)

    def compute(self, model_cache: ModelCache):
        """
        Compute feature samples.
        """
        for layer_idx, layer_cache in tqdm(model_cache.layers.items(), desc="Extracting samples"):
            layer_sample_set = LayerSampleSet(layer_idx)
            layer_sample_set.compute(layer_cache, model_cache.block_size)
            self.layers[layer_idx] = layer_sample_set

    def export(self, outdir: Path, shard: DatasetShard, tokenizer: Tokenizer):
        """
        Export feature samples to JSON
        """
        for layer_sample_set in self:
            for feature_sample_set in tqdm(
                layer_sample_set, desc=f"Exporting features from layer {layer_sample_set.layer_idx}"
            ):
                feature_outdir = outdir / f"{feature_sample_set.layer_idx}/{feature_sample_set.feature_idx}.json"
                feature_sample_set.export(feature_outdir, shard, tokenizer)

    def save(self, checkpoint_dir: Path):
        """
        Save feature samples to disk.
        """
        for layer_sample_set in self.layers.values():
            layer_sample_set.save(checkpoint_dir)

    def get_num_layers(self, checkpoint_dir: Path) -> int:
        """
        Count the number of files prefixed with "metrics.samples."
        """
        if count := len([f for f in checkpoint_dir.iterdir() if f.name.startswith("metrics.samples.")]):
            return count
        raise FileNotFoundError("Samples don't exist. Please run `compute_metrics`.")


class LayerSampleSet:
    """
    Contains feature samples for a layer.
    """

    SAMPLES_PERCENTILE = 0  # Percentile to use for feature samples
    SUGGESTED_SAMPLE_COUNT = 250  # Number of samples to use (if possible)

    def __init__(self, layer_idx: int):
        """
        Create empty layer sample set.
        """
        self.layer_idx = layer_idx
        self.sample_indices: dict[int, np.ndarray] = {}  # Indices with shape (num_samples,)
        self.sample_magnitudes: dict[int, sparse.csr_matrix] = {}  # Magnitudes with shape (num_samples, block_size)

    def __getitem__(self, feature_idx: int) -> "FeatureSampleSet":
        """
        Get the feature sample set for a given feature index.
        """
        sample_indices = self.sample_indices.get(feature_idx, np.array([]))
        sample_magnitudes = self.sample_magnitudes.get(feature_idx, sparse.csr_matrix((0, 0)))
        return FeatureSampleSet(self.layer_idx, feature_idx, sample_indices, sample_magnitudes)

    def __iter__(self) -> Iterator["FeatureSampleSet"]:
        """
        Return an iterator over the feature sample sets.
        """
        for feature_idx in self.sample_indices.keys():
            yield self[feature_idx]

    def __len__(self) -> int:
        """
        Return the number of feature sample sets.
        """
        return len(self.sample_indices)

    def compute(self, layer_cache: LayerCache, block_size: int):
        """
        Extract feature samples.
        """
        num_features = layer_cache.csc_matrix.shape[-1]  # type: ignore
        for feature_idx in range(num_features):
            # List of shard token indices to use for the feature sample
            sample_token_idxs: list[int]

            # Are there enough samples to use percentile thresholding?
            sparse_feature_magnitudes = layer_cache.csc_matrix[:, feature_idx]
            non_zero_rows = sparse_feature_magnitudes.nonzero()[0]
            non_zero_magnitudes = sparse_feature_magnitudes.data
            use_topk = non_zero_magnitudes.size < 100 / (100.0 - self.SAMPLES_PERCENTILE) * self.SUGGESTED_SAMPLE_COUNT
            if use_topk:
                topk = min(non_zero_magnitudes.size, self.SUGGESTED_SAMPLE_COUNT)
                top_rows = sorted(zip(non_zero_rows, non_zero_magnitudes), key=lambda x: -x[1])[:topk]
                sample_token_idxs = [int(idx) for idx, _ in top_rows]
            else:
                sample_token_idxs = []
                threshold = np.percentile(non_zero_magnitudes, self.SAMPLES_PERCENTILE)
                top_rows = [(idx, m) for (idx, m) in zip(non_zero_rows, non_zero_magnitudes) if m >= threshold]
                select_rows = sorted(random.sample(top_rows, self.SUGGESTED_SAMPLE_COUNT), key=lambda x: -x[1])
                sample_token_idxs = [int(idx) for idx, _ in select_rows]

            # Convert token indices to sample index ranges
            sample_starting_idxs = [idx - idx % block_size for idx in sample_token_idxs]
            sample_ending_idxs = [idx + block_size for idx in sample_starting_idxs]

            # Collect magnitudes for each sample
            sample_feature_magnitudes = []
            for starting_idx, ending_idx in zip(sample_starting_idxs, sample_ending_idxs):
                sparse_feature_magnitudes = layer_cache.csc_matrix[starting_idx:ending_idx, feature_idx]
                sparse_feature_magnitudes = sparse_feature_magnitudes.todense().reshape(1, -1)
                sample_feature_magnitudes.append(sparse_feature_magnitudes)

            # Store samples
            if len(sample_feature_magnitudes) > 0:
                sample_feature_magnitudes = sparse.csr_matrix(np.vstack(sample_feature_magnitudes))
                self.sample_indices[feature_idx] = np.array(sample_token_idxs)
                self.sample_magnitudes[feature_idx] = sample_feature_magnitudes

    def save(self, checkpoint_dir: Path):
        """
        Save feature samples to disk.
        """
        np.savez(
            checkpoint_dir / self.filename,
            **{f"{feature_idx}.indices": v for feature_idx, v in self.sample_indices.items()},  # type: ignore
            **{f"{feature_idx}.magnitudes-data": v.data for feature_idx, v in self.sample_magnitudes.items()},
            **{f"{feature_idx}.magnitudes-indices": v.indices for feature_idx, v in self.sample_magnitudes.items()},
            **{f"{feature_idx}.magnitudes-indptr": v.indptr for feature_idx, v in self.sample_magnitudes.items()},
            **{f"{feature_idx}.magnitudes-shape": v.shape for feature_idx, v in self.sample_magnitudes.items()},  # type: ignore
        )

    def load(self, checkpoint_dir: Path):
        """
        Load feature samples from disk.
        """
        with np.load(checkpoint_dir / self.filename, allow_pickle=True) as data:
            for key, value in data.items():
                match key.split("."):
                    case [feature_idx, "indices"]:
                        self.sample_indices[int(feature_idx)] = value
                    case [feature_idx, "magnitudes-data"]:
                        matrix_data = value
                        matrix_indices = data[f"{feature_idx}.magnitudes-indices"]
                        matrix_indptr = data[f"{feature_idx}.magnitudes-indptr"]
                        shape = data[f"{feature_idx}.magnitudes-shape"]
                        magnitudes = sparse.csr_matrix((matrix_data, matrix_indices, matrix_indptr), shape=shape)
                        self.sample_magnitudes[int(feature_idx)] = magnitudes
                    case _:
                        pass

    @property
    def filename(self) -> str:
        """
        Return the output file path.
        """
        return f"metrics.samples.{self.layer_idx}.npz"


@dataclass
class FeatureSampleSet:
    """
    Contains feature samples for a feature.
    """

    layer_idx: int
    feature_idx: int
    sample_indices: np.ndarray  # One shard token index for each sample. Shape: (num_samples,)
    sample_magnitudes: sparse.csr_matrix  # Shape: (num_samples, block_size)

    @cached_property
    def samples(self) -> list["Sample"]:
        """
        Return a list of feature samples.
        """
        feature_samples = []
        block_size: int = self.sample_magnitudes.shape[-1]  # type: ignore
        for i, shard_token_idx in enumerate(self.sample_indices):
            sample_idx = int(shard_token_idx // block_size)
            token_idx = int(shard_token_idx % block_size)
            sample = Sample(self.layer_idx, sample_idx, token_idx, self.sample_magnitudes[i])
            feature_samples.append(sample)
        return feature_samples

    def export(self, outdir: Path, shard: DatasetShard, tokenizer: Tokenizer):
        """
        Export feature samples to JSON
        """
        data = {
            "layer_idx": self.layer_idx,
            "feature_idx": self.feature_idx,
            "count": len(self.samples),
            "samples": [],
        }
        for sample in self.samples:
            block_size = sample.magnitudes.shape[-1]  # type: ignore
            starting_idx = sample.block_idx * block_size
            tokens = shard.tokens[starting_idx : starting_idx + block_size].tolist()
            data["samples"].append(
                {
                    "block_idx": sample.block_idx,
                    "token_idx": sample.token_idx,
                    "text": tokenizer.decode_sequence(tokens),
                    "tokens": tokens,
                    "magnitude_idxs": sample.magnitudes.indices.tolist(),
                    "magnitude_values": list(map(lambda x: round(x, 6), sample.magnitudes.data.tolist())),  # type: ignore
                }
            )
        outdir.parent.mkdir(parents=True, exist_ok=True)
        with open(outdir, "w") as f:
            f.write(json_prettyprint(data))


@dataclass
class Sample:
    """
    Contains magnitudes for a specific sample.
    """

    layer_idx: int
    block_idx: int  # Shard token idx // block_size
    token_idx: int  # Shard token idx % block_size
    magnitudes: sparse.csr_matrix  # Shape: (1, block_size)
