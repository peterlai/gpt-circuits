from typing import Protocol

import numpy as np
import torch

from circuits import Circuit, Node
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile


class Ablator(Protocol):
    """
    Ablation interface.
    """

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,
        circuit: Circuit,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Patch feature magnitudes and return `num_samples` samples.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param target_token_idx: Target token index for which logits are evaluated.
        :param feature_magnitudes: Feature magnitudes to patch. Shape: (T, F)
        :param circuit: Circuit representing features to preserve.
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, T, F)
        """
        ...


class ResampleAblator(Ablator):
    """
    Ablation using resampling. Based on technique described here:
    https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN
    """

    CacheKey = tuple[int, int, tuple[int, ...], tuple[float, ...]]  # Key type for caching nearest neighbors
    nearest_neighbors_cache: dict[CacheKey, np.ndarray] = {}  # Map of cached nearest neighbors

    def __init__(
        self,
        model_profile: ModelProfile,
        model_cache: ModelCache,
        k_nearest: int | None,
        positional_coefficient: float = 1.0,
    ):
        """
        :param model_cache: Model cache to use for resampling.
        :param k_nearest: Number of nearest neighbors to use for creating sample distributions. If `None`, use all.
        :param positional_coefficient: Coefficient for positional distance in MSE.
        """
        self.model_profile = model_profile
        self.model_cache = model_cache
        self.k_nearest = k_nearest
        self.positional_coefficient = positional_coefficient

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,  # Shape: (T, F)
        circuit: Circuit,
        num_samples: int,
    ) -> torch.Tensor:  # Shape: (B, T, F)
        """
        Resample feature magnitudes using cached values, returning `num_samples` samples.
        Samples are drawn from `k_nearest` most similar rows in the cache.
        """
        # Construct empty samples
        samples_shape = (num_samples,) + feature_magnitudes.shape
        samples = torch.zeros(samples_shape, device=feature_magnitudes.device)

        # Ignore tokens after the target token because they'll be ignored.
        for token_idx in range(target_token_idx + 1):
            token_idx, token_samples = self.patch_token_magnitudes(
                layer_idx,
                token_idx,
                feature_magnitudes[token_idx].cpu().numpy(),  # Shape: (F)
                {node for node in circuit.nodes if node.token_idx == token_idx and node.layer_idx == layer_idx},
                num_samples,
            )
            samples[:, token_idx, :] = token_samples

        return samples

    def patch_token_magnitudes(
        self,
        layer_idx: int,
        token_idx: int,
        token_feature_magnitudes: np.ndarray,
        token_nodes: set[Node],
        num_samples: int,
    ) -> tuple[int, torch.Tensor]:
        """
        Patch feature magnitudes for a single token.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param token_idx: Token index for which features are sampled.
        :param token_feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param token_nodes: Circuit nodes within a token representing features to preserve.
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, F)
        """
        layer_cache = self.model_cache[layer_idx]
        circuit_feature_idxs = np.array([f.feature_idx for f in token_nodes]).astype(np.int32)

        if self.k_nearest is not None:
            # Get nearest neighbors using MSE
            nearest_neighbor_idxs = self.get_nearest_neighbors(
                layer_idx,
                token_idx,
                token_feature_magnitudes,
                circuit_feature_idxs,
            )
        else:
            # Randomly sample from the layer cache
            num_shard_tokens: int = layer_cache.magnitudes.shape[0]  # type: ignore
            sequence_idxs = np.random.choice(
                range(num_shard_tokens // layer_cache.block_size),
                size=num_samples,
                replace=False,
            )
            # Respect token position when choosing indices
            nearest_neighbor_idxs = sequence_idxs * layer_cache.block_size + token_idx

        # Randomly draw indices from top candidates to include in samples
        sample_size = min(len(nearest_neighbor_idxs), num_samples)
        sample_idxs = np.random.choice(nearest_neighbor_idxs, size=sample_size, replace=False)
        # If there are too few candidates, duplicate some
        if len(sample_idxs) < num_samples:
            num_duplicates = num_samples - len(sample_idxs)
            extra_sample_idxs = np.random.choice(sample_idxs, size=num_duplicates, replace=True)
            sample_idxs = np.concatenate((sample_idxs, extra_sample_idxs))
        token_samples = layer_cache.csr_matrix[sample_idxs, :].toarray()  # Shape: (B, F)

        # Preserve circuit feature magnitudes
        token_samples[:, circuit_feature_idxs] = token_feature_magnitudes[circuit_feature_idxs]

        # Return token index and patched feature magnitudes
        return token_idx, torch.tensor(token_samples)

    def get_nearest_neighbors(
        self,
        layer_idx: int,
        token_idx: int,
        feature_magnitudes: np.ndarray,  # Shape: (F)
        circuit_feature_idxs: np.ndarray,
    ) -> np.ndarray:
        """
        Get nearest neighbors for a single token in a given circuit.

        :param layer_cache: Layer cache to use for resampling.
        :param token_idx: Token index for which features are sampled.
        :param feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param circuit_feature_idxs: Indices of features to preserve for this token.

        :return: Indices of nearest neighbors.
        """
        assert self.k_nearest is not None
        layer_profile = self.model_profile[layer_idx]
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size
        num_features = len(circuit_feature_idxs)

        # Check if nearest neighbors are cached
        # TODO: Consider purging unused cache entries
        circuit_feature_magnitudes = feature_magnitudes[circuit_feature_idxs]
        cache_key = self.get_cache_key(layer_idx, token_idx, circuit_feature_idxs, circuit_feature_magnitudes)
        if cache_key in self.nearest_neighbors_cache:
            return self.nearest_neighbors_cache[cache_key]

        if num_features == 0:
            # No features to use for sampling
            top_feature_idxs = np.array([])
        else:
            # Get top features by magnitude
            num_top_features = max(0, min(16, num_features))  # Limit to 16 features
            # TODO: Consider selecting top features using normalized magnitude
            select_indices = np.argsort(circuit_feature_magnitudes)[-num_top_features:]
            top_feature_idxs = circuit_feature_idxs[select_indices]

        # Find rows in layer cache with any of the top features
        row_idxs = np.unique(layer_cache.csc_matrix[:, top_feature_idxs].nonzero()[0])

        # Use random rows selected based on token position if no feature-based candidates
        if len(row_idxs) == 0:
            num_blocks = layer_cache.magnitudes.shape[0] // block_size  # type: ignore
            sample_idxs = np.random.choice(range(num_blocks), size=self.k_nearest, replace=True)
            row_idxs = sample_idxs * block_size + token_idx

        # Create matrix of token magnitudes to sample from
        candidate_samples = layer_cache.csc_matrix[:, circuit_feature_idxs][row_idxs, :].toarray()
        target_values = feature_magnitudes[circuit_feature_idxs]

        # Calculate normalization coefficients
        norm_coefficients = np.ones_like(target_values)
        for i, feature_idx in enumerate(circuit_feature_idxs):
            feature_profile = layer_profile[int(feature_idx)]
            norm_coefficients[i] = 1.0 / feature_profile.max

        # Add positional information
        positional_distances = np.abs((row_idxs % block_size) - token_idx).astype(np.float32)
        positional_distances = positional_distances / block_size  # Scale to [0, 1]
        candidate_samples = np.column_stack((candidate_samples, positional_distances))  # Add column
        target_values = np.append(target_values, 0)  # Add target
        norm_coefficients = np.append(norm_coefficients, self.positional_coefficient)  # Add coefficient

        # Calculate MSE
        errors = (candidate_samples - target_values) * norm_coefficients
        mse = np.mean(errors**2, axis=-1)

        # Get nearest neighbors
        num_neighbors = min(self.k_nearest, len(row_idxs))
        # TODO: Consider removing first exact match to avoid duplicating original values
        nearest_neighbor_idxs = row_idxs[np.argsort(mse)[:num_neighbors]]

        # Cache nearest neighbors
        self.nearest_neighbors_cache[cache_key] = nearest_neighbor_idxs
        return nearest_neighbor_idxs

    def get_cache_key(
        self,
        layer_idx: int,
        token_idx: int,
        circuit_feature_idxs: np.ndarray,
        circuit_feature_magnitudes: np.ndarray,
    ) -> CacheKey:
        """
        Get cache key for nearest neighbors.
        """
        return (
            layer_idx,
            token_idx,
            tuple([int(f) for f in circuit_feature_idxs]),
            tuple([float(f) for f in circuit_feature_magnitudes]),
        )


class ZeroAblator(Ablator):
    """
    Ablation using zeroing of patched features.
    """

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: torch.Tensor,
        circuit: Circuit,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Set non-circuit features to zero and duplicate the result S times. (Useful for debugging)
        """
        # Zero-ablate non-circuit features
        patched_feature_magnitudes = torch.zeros_like(feature_magnitudes)
        if circuit.nodes:
            token_idxs: list[int] = []
            feature_idxs: list[int] = []
            for node in circuit.nodes:
                # Only preserve features from the target layer that are on or before the target token
                if node.layer_idx == layer_idx and node.token_idx <= target_token_idx:
                    token_idxs.append(node.token_idx)
                    feature_idxs.append(node.feature_idx)
            patched_feature_magnitudes[token_idxs, feature_idxs] = feature_magnitudes[token_idxs, feature_idxs]

        # Duplicate feature magnitudes `num_samples` times
        patched_samples = patched_feature_magnitudes.unsqueeze(0).expand(num_samples, -1, -1)  # Shape: (B, T, F)
        return patched_samples
