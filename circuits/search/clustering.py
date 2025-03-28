from dataclasses import dataclass

import numpy as np
from scipy import sparse

from circuits.features.cache import LayerCache, ModelCache
from circuits.features.profiles import ModelProfile
from circuits.features.samples import Sample


@dataclass(frozen=True)
class ClusterCacheKey:
    """
    Cache key for clustering results.
    """

    layer_idx: int
    token_idx: int
    circuit_feature_idxs: tuple[int, ...]
    circuit_feature_magnitudes: tuple[float, ...]
    k_nearest: int
    feature_coefficients: tuple[float, ...]
    positional_coefficient: float


class Cluster:
    """
    Cluster of nearest neighbors using feature magnitudes for a token in a given circuit.
    """

    layer_cache: LayerCache
    layer_idx: int
    token_idx: int
    idxs: tuple[int, ...]
    mses: tuple[float, ...]

    def __init__(
        self,
        layer_cache: LayerCache,
        layer_idx: int,
        token_idx: int,
        idxs: tuple[int, ...],
        mses: tuple[float, ...],
    ):
        self.layer_cache = layer_cache
        self.layer_idx = layer_idx
        self.token_idx = token_idx
        self.idxs = idxs
        self.mses = mses

    def sample_magnitudes(self, num_samples: int) -> sparse.csr_matrix:
        """
        Sample feature magnitudes from the cluster.

        :param num_samples: Number of samples to return.

        :return: Sampled feature magnitudes. Shape: (num_samples, F)
        """
        sample_size = min(len(self.idxs), num_samples)
        sample_idxs = np.random.choice(self.idxs, size=sample_size, replace=False)

        # If there are too few candidates, duplicate some
        if len(sample_idxs) < num_samples:
            num_duplicates = num_samples - len(sample_idxs)
            extra_sample_idxs = np.random.choice(sample_idxs, size=num_duplicates, replace=True)
            sample_idxs = np.concatenate((sample_idxs, extra_sample_idxs))

        # Get feature magnitudes
        feature_magnitudes = self.layer_cache.csr_matrix[sample_idxs, :].toarray()  # Shape: (num_samples, F)
        return feature_magnitudes


class ClusterSampleSet:
    """
    Set of samples representing a cluster of nearest neighbors.
    """

    samples: list[Sample]

    def __init__(self, cluster: Cluster, sample_magnitudes: list[sparse.csr_matrix]):
        """
        Construct a sample set from a cluster of nearest neighbors.

        :param cluster: Cluster of nearest neighbors.
        :param sample_magnitudes: List of feature magnitudes for each sample.
        :return: SampleSet representing the cluster.
        """
        block_size = cluster.layer_cache.block_size

        self.samples = []
        for shard_token_idx, magnitudes in zip(cluster.idxs, sample_magnitudes):
            block_idx = shard_token_idx // block_size
            token_idx = shard_token_idx % block_size
            self.samples.append(
                Sample(
                    layer_idx=cluster.layer_idx,
                    block_idx=block_idx,
                    token_idx=token_idx,
                    magnitudes=magnitudes,
                )
            )


class ClusterSearch:
    """
    Search for nearest neighbors using cached feature magnitudes.
    """

    # Map of cached nearest neighbors
    cached_cluster_idxs: dict[ClusterCacheKey, tuple[int, ...]] = {}

    def __init__(self, model_profile: ModelProfile, model_cache: ModelCache):
        self.model_profile = model_profile
        self.model_cache = model_cache

    def get_cluster(
        self,
        layer_idx: int,
        token_idx: int,
        feature_magnitudes: np.ndarray,  # Shape: (F)
        circuit_feature_idxs: np.ndarray,
        k_nearest: int,
        feature_coefficients: np.ndarray,  # Length must match len(circuit_feature_idxs)
        positional_coefficient: float,
    ) -> Cluster:
        """
        Get nearest neighbors for a single token in a given circuit.

        :param layer_idx: Layer index from which features are sampled.
        :param token_idx: Token index from which features are sampled.
        :param feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param circuit_feature_idxs: Indices of features to preserve for this token.
        :param k_nearest: Number of nearest neighbors to return.
        :param feature_coefficients: Coefficients representing the importance of each circuit feature.
        :param positional_coefficient: Coefficient representing the importance of positional information.

        :return: Cluster representing nearest neighbor.
        """
        assert k_nearest > 0
        num_features = len(circuit_feature_idxs)

        # Get candidate indices if there are any features to preserve
        if num_features > 0:
            layer_cache = self.model_cache[layer_idx]

            # Check if nearest neighbors are cached
            # TODO: Consider purging unused cache entries
            circuit_feature_magnitudes = feature_magnitudes[circuit_feature_idxs]
            cache_key = self.get_cache_key(
                layer_idx,
                token_idx,
                circuit_feature_idxs,
                circuit_feature_magnitudes,
                k_nearest,
                feature_coefficients,
                positional_coefficient,
            )
            if cluster_idxs := self.cached_cluster_idxs.get(cache_key):
                return Cluster(
                    layer_cache=layer_cache,
                    layer_idx=layer_idx,
                    token_idx=token_idx,
                    idxs=cluster_idxs,
                    mses=(0.0,) * len(cluster_idxs),
                )

            # Get top features by magnitude to use for narrowing down candidates
            # TODO: Consider selecting top features using normalized magnitude
            num_top_features = max(0, min(16, num_features))  # Limit to 16 features for performance
            select_indices = np.argsort(circuit_feature_magnitudes)[-num_top_features:]
            top_feature_idxs = circuit_feature_idxs[select_indices]

            # Find rows in layer cache with any of the top features
            candidate_row_idxs = np.unique(layer_cache.csc_matrix[:, top_feature_idxs].nonzero()[0])

            # If candidates exist, get cluster using candidate indices
            if len(candidate_row_idxs) > 0:
                # Use all features for calculating MSE
                target_feature_idxs = circuit_feature_idxs
                target_feature_values = feature_magnitudes[circuit_feature_idxs]
                cluster_idxs, cluster_mses = self.get_cluster_from_candidate_idxs(
                    layer_idx,
                    token_idx,
                    candidate_row_idxs,
                    target_feature_idxs,
                    target_feature_values,
                    feature_coefficients,
                    positional_coefficient,
                    k_nearest,
                )
                cluster = Cluster(
                    layer_cache=layer_cache,
                    layer_idx=layer_idx,
                    token_idx=token_idx,
                    idxs=cluster_idxs,
                    mses=cluster_mses,
                )
                # Cache cluster before returning it
                self.cached_cluster_idxs[cache_key] = cluster.idxs
                return cluster

        # Fallback to random sampling
        return self.get_random_cluster(
            layer_idx,
            token_idx,
            num_samples=k_nearest,
            positional_coefficient=positional_coefficient,
        )

    def get_cluster_as_sample_set(
        self,
        layer_idx: int,
        token_idx: int,
        feature_magnitudes: np.ndarray,  # Shape: (F)
        circuit_feature_idxs: np.ndarray,
        k_nearest: int,
        feature_coefficients: np.ndarray,  # Length must match len(circuit_feature_idxs)
        positional_coefficient: float,
    ) -> ClusterSampleSet:
        """
        Get nearest neighbors for a single token in a given circuit as a sample set.

        :return: ClusterSampleSet representing nearest neighbors.
        """
        cluster = self.get_cluster(
            layer_idx,
            token_idx,
            feature_magnitudes,
            circuit_feature_idxs,
            k_nearest,
            feature_coefficients,
            positional_coefficient,
        )

        # Prepare targets
        target_feature_idxs = circuit_feature_idxs
        target_feature_values = feature_magnitudes[circuit_feature_idxs]

        # Get shard token indices to use for calculating MSE
        relative_token_range = list(range(-16, 17))
        shard_token_idxs = set()
        for shard_token_idx in cluster.idxs:
            shard_token_idxs.update([shard_token_idx + i for i in relative_token_range])

        # Filter out negative indices and indices beyond the number of tokens in the shard
        num_tokens = cluster.layer_cache.magnitudes.shape[0]  # type: ignore
        shard_token_idxs = set(filter(lambda x: 0 <= x < num_tokens, shard_token_idxs))
        shard_token_idxs = np.array(list(sorted(shard_token_idxs)))

        # Map shard token indices to MSEs that ignore positional information
        # NOTE: Positional information makes interpreting the sample magnitudes more difficult
        mse_idxs, mse_values = self.get_cluster_from_candidate_idxs(
            layer_idx,
            token_idx,
            shard_token_idxs,
            target_feature_idxs,
            target_feature_values,
            feature_coefficients,
            positional_coefficient=0.0,
            k_nearest=len(shard_token_idxs),  # Get MSE for all shard token indices
        )
        shard_token_idx_to_mse = dict(zip(mse_idxs, mse_values))

        # Caculate max MSE using percentile
        # We want at most 75% of the samples to have a non-zero magnitude
        max_mse = np.percentile(np.array(list(shard_token_idx_to_mse.values())), 75).item()

        # Calculate sample magnitudes
        block_size = cluster.layer_cache.block_size
        sample_magnitudes: list[sparse.csr_matrix] = []
        for shard_token_idx in cluster.idxs:
            block_idx = shard_token_idx // block_size
            token_idx = shard_token_idx % block_size

            # Calculate magnitudes for each token in the relative range
            magnitudes = np.zeros(shape=(1, block_size))
            for offset in relative_token_range:
                adjusted_token_idx = token_idx + offset
                if 0 <= adjusted_token_idx < block_size:
                    adjusted_shard_token_idx = block_idx * block_size + adjusted_token_idx
                    mse = shard_token_idx_to_mse.get(adjusted_shard_token_idx, max_mse)
                    divisor = max(1e-10, max_mse)  # Avoid division by zero
                    magnitude = max(1.0 - mse / divisor, 0)  # Avoid negative values
                    magnitude = magnitude**2  # Square to emphasize differences
                    magnitude = magnitude if magnitude > 0.1 else 0.0  # Ignore small values
                    magnitudes[0, adjusted_token_idx] = magnitude

            magnitudes = sparse.csr_matrix(magnitudes)
            sample_magnitudes.append(magnitudes)

        return ClusterSampleSet(cluster, sample_magnitudes)

    def get_cluster_from_candidate_idxs(
        self,
        layer_idx: int,
        token_idx: int,
        candidate_row_idxs: np.ndarray,
        target_feature_idxs: np.ndarray,
        target_feature_values: np.ndarray,
        feature_coefficients: np.ndarray,
        positional_coefficient: float,
        k_nearest: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """
        Get nearest neighbors from candidate indices.

        :param layer_idx: Layer index from which features are sampled.
        :param token_idx: Token index to use for positional distance.
        :param candidate_row_idxs: Indices of candidate rows in the layer cache.
        :param target_feature_idxs: Indices of features to preserve for this token.
        :param target_feature_values: Values of features to preserve for this token.
        :param feature_coefficients: Coefficients representing the importance of each circuit feature.
        :param positional_coefficient: Coefficient representing the importance of positional information.
        :param k_nearest: Number of nearest neighbors to return.

        :return: Tuple representing nearest neighbors:
            - cluster_idxs: Indices of nearest neighbors.
            - cluster_mses: Mean squared errors of nearest neighbors.
        """
        assert len(target_feature_idxs) == len(feature_coefficients), "Each coefficient must correspond to an idx"
        assert len(target_feature_idxs) == len(target_feature_values), "Each feature value must correspond to an idx"
        layer_profile = self.model_profile[layer_idx]
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size

        # Create matrix of token magnitudes to sample from
        candidate_samples = layer_cache.csc_matrix[:, target_feature_idxs][candidate_row_idxs, :].toarray()

        # Calculate normalization coefficients
        norm_coefficients = np.ones_like(target_feature_values)
        for i, feature_idx in enumerate(target_feature_idxs):
            feature_profile = layer_profile[int(feature_idx)]
            norm_coefficients[i] = 1.0 / feature_profile.max

        # Add positional information
        positional_distances = np.abs((candidate_row_idxs % block_size) - token_idx).astype(np.float32)
        positional_distances = positional_distances / block_size  # Scale to [0, 1]
        candidate_samples = np.column_stack((candidate_samples, positional_distances))  # Add column
        target_feature_values = np.append(target_feature_values, 0.0)  # Add target
        norm_coefficients = np.append(norm_coefficients, 1.0)

        # Calculate MSE
        multipliers = np.append(feature_coefficients, positional_coefficient)  # How important is each dimension?
        squared_errors = ((candidate_samples - target_feature_values) * norm_coefficients * multipliers) ** 2
        mses = np.mean(squared_errors, axis=-1)

        # Get nearest neighbors
        if k_nearest < len(candidate_row_idxs):
            partition_idxs = np.argpartition(mses, k_nearest)[:k_nearest]
        else:
            partition_idxs = np.arange(len(candidate_row_idxs))
        cluster_idxs: tuple[int, ...] = tuple(candidate_row_idxs[partition_idxs].tolist())  # type: ignore
        cluster_mses = tuple(mses[partition_idxs].tolist())

        # Return cluster
        return cluster_idxs, cluster_mses

    def get_random_cluster(
        self,
        layer_idx: int,
        token_idx: int,
        num_samples: int,
        positional_coefficient: float,
    ) -> Cluster:
        """
        Get random cluster for a given layer and token position.

        :param layer_idx: Layer index from which features are sampled.
        :param token_idx: Token index from which features are sampled.
        :param num_samples: Number of samples to include in the cluster.
        :param positional_coefficient: Coefficient representing the importance of positional information.

        :return: Cluster representing random samples.
        """
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size
        num_shard_tokens: int = layer_cache.magnitudes.shape[0]  # type: ignore

        # Choose cluster indices
        block_idxs = np.random.choice(range(num_shard_tokens // block_size), size=num_samples, replace=False)
        if positional_coefficient > 0.0:
            # If positional information is important, respect token position when choosing indices
            token_idxs = np.full_like(block_idxs, token_idx)
        else:
            # Else, choose random token positions
            token_idxs = np.random.choice(range(block_size), size=num_samples, replace=True)
        cluster_idxs = block_idxs * layer_cache.block_size + token_idxs
        cluster_mses = (0.0,) * len(cluster_idxs)
        return Cluster(
            layer_cache=layer_cache,
            layer_idx=layer_idx,
            token_idx=token_idx,
            idxs=cluster_idxs,
            mses=cluster_mses,
        )

    def get_cache_key(
        self,
        layer_idx: int,
        token_idx: int,
        circuit_feature_idxs: np.ndarray,
        circuit_feature_magnitudes: np.ndarray,
        k_nearest: int,
        feature_coefficients: np.ndarray,
        positional_coefficient: float,
    ) -> ClusterCacheKey:
        """
        Get cache key for nearest neighbors.
        """
        return ClusterCacheKey(
            layer_idx,
            token_idx,
            tuple([int(f) for f in circuit_feature_idxs]),
            tuple([float(f) for f in circuit_feature_magnitudes]),
            k_nearest,
            tuple([float(f) for f in feature_coefficients]),
            positional_coefficient,
        )
