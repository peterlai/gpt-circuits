from typing import Protocol

import numpy as np
import torch

from circuits import Circuit, Node
from circuits.features.cache import ModelCache
from circuits.features.profiles import ModelProfile
from circuits.search.clustering import ClusterSearch


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
        self.cluster_search = ClusterSearch(model_profile, model_cache)

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
        circuit_feature_idxs = np.array([f.feature_idx for f in token_nodes]).astype(np.int32)

        if self.k_nearest is not None:
            # Get cluster representing nearest neighbors
            cluster = self.cluster_search.get_cluster(
                layer_idx,
                token_idx,
                token_feature_magnitudes,
                circuit_feature_idxs,
                k_nearest=self.k_nearest,
                feature_coefficients=np.ones_like(circuit_feature_idxs),
                positional_coefficient=self.positional_coefficient,
            )
        else:
            # Create a random cluster using tokens with the same token position
            cluster = self.cluster_search.get_random_cluster(layer_idx, token_idx, num_samples)

        # Randomly draw sample magnitudes from the cluster
        token_samples = cluster.sample_magnitudes(num_samples)

        # Preserve circuit feature magnitudes
        token_samples[:, circuit_feature_idxs] = token_feature_magnitudes[circuit_feature_idxs]

        # Return token index and patched feature magnitudes
        return token_idx, torch.tensor(token_samples)


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
