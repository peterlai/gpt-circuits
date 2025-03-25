import numpy as np

from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.clustering import ClusterSearch


class ResampleAblator:
    """
    Ablation using resampling. Based on technique described here:
    https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN
    """

    def __init__(
        self,
        model_profile: ModelProfile,
        model_cache: ModelCache,
        k_nearest: int | None,
        positional_coefficient: float = 1.0,
    ):
        """
        :param model_cache: Model cache to use for resampling.
        :param k_nearest: Number of nearest neighbors to use for creating sample distributions.
            - If `None`, use random sampling.
            - If `0`, use zero ablation.
        :param positional_coefficient: Coefficient for positional distance in MSE.
        """
        self.model_profile = model_profile
        self.model_cache = model_cache
        self.k_nearest = k_nearest
        self.positional_coefficient = positional_coefficient
        self.cluster_search = ClusterSearch(model_profile, model_cache)

    def patch(
        self,
        layer_idx: int,
        target_token_idx: int,
        feature_magnitudes: np.ndarray,  # Shape: (T, F)
        feature_mask: np.ndarray,  # Shape: (T, F)
        num_samples: int,
    ) -> np.ndarray:  # Shape: (B, T, F)
        """
        Resample feature magnitudes using cached values, returning `num_samples` samples.
        Samples are drawn from `k_nearest` most similar rows in the cache.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param target_token_idx: Target token index for which logits are evaluated.
        :param feature_magnitudes: Feature magnitudes to patch. Shape: (T, F)
        :param feature_mask: Features magnitudes to preserve. Shape: (T, F)
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, T, F)
        """
        # Construct empty samples
        samples_shape = (num_samples,) + feature_magnitudes.shape
        samples = np.zeros(samples_shape, dtype=np.float32)

        # Ignore tokens after the target token because they'll be ignored.
        for token_idx in range(target_token_idx + 1):
            token_idx, token_samples = self.patch_token_magnitudes(
                layer_idx,
                token_idx,
                feature_magnitudes[token_idx],  # Shape: (F)
                feature_mask[token_idx],  # Shape: (F)
                num_samples,
            )
            samples[:, token_idx, :] = token_samples

        return samples

    def patch_token_magnitudes(
        self,
        layer_idx: int,
        token_idx: int,
        token_feature_magnitudes: np.ndarray,
        token_feature_mask: np.ndarray,
        num_samples: int,
    ) -> tuple[int, np.ndarray]:
        """
        Patch feature magnitudes for a single token.

        :param layer_idx: Layer index from which feature magnitudes are taken.
        :param token_idx: Token index for which features are sampled.
        :param token_feature_magnitudes: Feature magnitudes for the token. Shape: (F)
        :param token_feature_mask: Features to preserve. Shape: (F)
        :param num_samples: Number of samples to return.

        :return: Patched feature magnitudes. Shape: (B, F)
        """
        # If mask is all ones, return original magnitudes (repeated `num_samples` times)
        if np.all(token_feature_mask):
            return token_idx, np.tile(token_feature_magnitudes, (num_samples, 1))

        # Set the importance of each feature
        circuit_feature_idxs = np.where(token_feature_mask)[0].astype(np.int32)
        feature_coefficients = np.ones_like(circuit_feature_idxs)

        match self.k_nearest:
            # Conventional resampling
            case None:
                # Create a random cluster
                cluster = self.cluster_search.get_random_cluster(
                    layer_idx,
                    token_idx,
                    num_samples,
                    self.positional_coefficient,
                )

                # Randomly draw sample magnitudes from the cluster
                token_samples = cluster.sample_magnitudes(num_samples)

            # Zero ablation
            case 0:
                token_samples = np.zeros(
                    (num_samples, len(token_feature_magnitudes)),
                    dtype=token_feature_magnitudes.dtype,
                )

            # Cluster resampling
            case _:
                # Get cluster representing nearest neighbors
                cluster = self.cluster_search.get_cluster(
                    layer_idx,
                    token_idx,
                    token_feature_magnitudes,
                    circuit_feature_idxs,
                    k_nearest=self.k_nearest,
                    feature_coefficients=feature_coefficients,
                    positional_coefficient=self.positional_coefficient,
                )

                # Randomly draw sample magnitudes from the cluster
                token_samples = cluster.sample_magnitudes(num_samples)

        # Preserve circuit feature magnitudes
        token_samples[:, circuit_feature_idxs] = token_feature_magnitudes[circuit_feature_idxs]

        # Return token index and patched feature magnitudes
        return token_idx, token_samples
