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
    max_mse: float  # Max MSE that is possible based on the coefficients used as multipliers

    def __init__(
        self,
        layer_cache: LayerCache,
        layer_idx: int,
        token_idx: int,
        idxs: tuple[int, ...],
        mses: tuple[float, ...],
        max_mse: float,
    ):
        self.layer_cache = layer_cache
        self.layer_idx = layer_idx
        self.token_idx = token_idx
        self.idxs = idxs
        self.mses = mses
        self.max_mse = max_mse

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

    def as_sample_set(self) -> "ClusterSampleSet":
        """
        Return all nearest neighbors as a sample set.
        """
        return ClusterSampleSet(self)


class ClusterSampleSet:
    """
    Set of samples representing a cluster of nearest neighbors.
    """

    samples: list[Sample]

    def __init__(self, cluster: Cluster):
        block_size = cluster.layer_cache.block_size

        self.samples = []
        for shard_token_idx, mse in zip(cluster.idxs, cluster.mses):
            block_idx = shard_token_idx // block_size
            token_idx = shard_token_idx % block_size

            # TODO: Consider better ways of normalizing magnitudes
            magnitudes = np.zeros(shape=(1, block_size))
            divisor = max(1e-10, cluster.max_mse)  # Avoid division by zero
            magnitudes[0, token_idx] = 1.0 - mse / divisor
            magnitudes = sparse.csr_matrix(magnitudes)

            sample = Sample(
                layer_idx=cluster.layer_idx,
                block_idx=block_idx,
                token_idx=token_idx,
                magnitudes=magnitudes,
            )
            self.samples.append(sample)


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

            # Calculate max MSE by averaging the squares of all coefficients
            max_mse = np.append(feature_coefficients**2, positional_coefficient**2).mean()

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
                    max_mse=max_mse,
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
                    max_mse=max_mse,
                )
                # Cache cluster before returning it
                self.cached_cluster_idxs[cache_key] = cluster.idxs
                return cluster

        # Fallback to random sampling
        return self.get_random_cluster(
            layer_idx,
            token_idx,
            num_samples=k_nearest,
            feature_coefficients=feature_coefficients,
            positional_coefficient=positional_coefficient,
        )

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
        num_neighbors = min(k_nearest, len(candidate_row_idxs))
        partition_idxs = np.argpartition(mses, num_neighbors)[:num_neighbors]
        cluster_idxs: tuple[int, ...] = tuple(candidate_row_idxs[partition_idxs].tolist())  # type: ignore
        cluster_mses = tuple(mses[partition_idxs].tolist())

        # Return cluster
        return cluster_idxs, cluster_mses

    def get_random_cluster(
        self,
        layer_idx: int,
        token_idx: int,
        num_samples: int,
        feature_coefficients: np.ndarray,
        positional_coefficient: float,
    ) -> Cluster:
        """
        Get random cluster for a given layer and token position.

        :param layer_idx: Layer index from which features are sampled.
        :param token_idx: Token index from which features are sampled.
        :param num_samples: Number of samples to include in the cluster.
        :param feature_coefficients: Coefficients representing the importance of each circuit feature.
        :param positional_coefficient: Coefficient representing the importance of positional information.

        :return: Cluster representing random samples.
        """
        layer_cache = self.model_cache[layer_idx]
        block_size = layer_cache.block_size
        num_shard_tokens: int = layer_cache.magnitudes.shape[0]  # type: ignore

        # Calculate max MSE by averaging the squares of all coefficients
        max_mse = np.append(feature_coefficients**2, positional_coefficient**2).mean()

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
            max_mse=max_mse,
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
