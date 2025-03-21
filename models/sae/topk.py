from typing import Optional

import torch
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class TopKSAE(nn.Module, SparseAutoencoder):
    """
    Top-k sparse autoencoder as described in:
    https://arxiv.org/pdf/2406.04093v1
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super(TopKSAE, self).__init__()
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size)))
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        assert config.top_k is not None, "checkpoints/<model_name>/sae.json must contain a 'top_k' key."
        self.k = config.top_k[layer_idx]

        try:
            # NOTE: Subclass might define these properties.
            self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size))
            self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        except KeyError:
            pass

        # Top-k SAE losses do not depend upon any loss coefficients; however, if an empty class is provided,
        # we know that we should compute losses and omit doing so otherwise.
        self.should_return_losses = loss_coefficients is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        latent = (x - self.b_dec) @ self.W_enc + self.b_enc

        # Zero out all but the top-k activations
        top_k_values, _ = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()

        return latent_k_sparse

    def decode(self, feature_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        feature_magnitudes: SAE activations (B, T, feature size)
        """
        return feature_magnitudes @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (B, T, embedding size)
        """
        feature_magnitudes = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)
        if self.should_return_losses:
            sparsity_loss = torch.tensor(0.0, device=x.device)  # no need for sparsity loss for top-k SAE
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output
