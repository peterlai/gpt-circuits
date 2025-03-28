from typing import Optional

import torch
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class StandardSAE(SparseAutoencoder):
    """
    SAE technique as described in:
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        super().__init__(layer_idx, config, loss_coefficients, model)
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        l1_coefficients = loss_coefficients.sparsity if loss_coefficients else None
        self.l1_coefficient = l1_coefficients[layer_idx] if l1_coefficients else None
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size)))

        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))

        try:
            # NOTE: Subclass might define these properties.
            self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size))
            self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        except KeyError:
            pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        return nn.ReLU()((x - self.b_dec) @ self.W_enc + self.b_enc)

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

        if self.l1_coefficient:
            decoder_norm = torch.norm(self.W_dec, p=2, dim=1)  # L2 norm
            sparsity_loss = (feature_magnitudes * decoder_norm).sum(dim=-1).mean() * self.l1_coefficient
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output


class StandardSAE_V2(StandardSAE):
    """
    Experimental Sparse Autoencoder module that ties the encoder and decoder weights to avoid feature absorption.
    Reference: https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN
    """

    @property
    def W_enc(self):
        """
        Tying encoder weights to decoder weights.
        """
        return self.W_dec.t()
