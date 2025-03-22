from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class GatedSAE(SparseAutoencoder):
    """
    Gated sparse autoencoder with RI-L1 sparsity penalty
    https://arxiv.org/abs/2404.16014
    https://arxiv.org/html/2407.14435v3
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__(layer_idx, config, loss_coefficients)
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        l1_coefficients = loss_coefficients.sparsity if loss_coefficients else None
        self.l1_coefficient = l1_coefficients[layer_idx] if l1_coefficients else None
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size)))
        self.b_gate = nn.Parameter(torch.zeros(feature_size))
        self.b_mag = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))

        try:
            # NOTE: Subclass might define these properties.
            self.W_gate = nn.Parameter(self.W_dec.mT.detach().clone())
            self.r_mag = nn.Parameter(torch.zeros(feature_size))
        except KeyError:
            pass

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: GPT model activations (B, T, embedding size)

        Implementation adapted from: https://github.com/saprmarks/dictionary_learning/
        """
        x_centered = x - self.b_dec
        x_enc = x_centered @ self.W_gate

        # Gating network
        pi_gate = x_enc + self.b_gate
        f_gate = (pi_gate > 0).to(self.W_gate.dtype)  # whether to gate the feature

        # Magnitude network
        pi_mag = x_enc * torch.exp(self.r_mag) + self.b_mag
        f_mag = F.relu(pi_mag)  # feature magnitudes

        feature_magnitudes = f_gate * f_mag
        return feature_magnitudes, pi_gate

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
        feature_magnitudes, pi_gate = self.encode(x)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)

        if self.l1_coefficient:
            # Use Gated (RI-L1) sparsity variant: https://arxiv.org/pdf/2407.14435
            scaled_pi_gate = F.relu(pi_gate) * self.W_dec.data.norm(dim=1)
            sparsity_loss = torch.norm(scaled_pi_gate, p=1, dim=-1).mean() * self.l1_coefficient

            # compute the auxiliary loss
            W_dec_clone = self.W_dec.clone().detach()
            b_dec_clone = self.b_dec.clone().detach()
            x_hat_frozen = nn.ReLU()(pi_gate) @ W_dec_clone + b_dec_clone
            aux_loss = (x - x_hat_frozen).pow(2).sum(dim=-1).mean()

            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss, aux_loss)

        return output


class GatedSAE_V2(GatedSAE):
    """
    Experimental Gated Sparse Autoencoder module that ties the encoder and decoder weights to avoid feature absorption.
    Reference: https://www.lesswrong.com/posts/kcg58WhRxFA9hv9vN
    """

    @property
    def W_gate(self):
        """
        Tying encoder weights to decoder weights.
        """
        return self.W_dec.t()

    @property
    def r_mag(self):
        """
        The r_mag scaler doesn't seem useful after weights are tied.
        """
        return torch.zeros_like(self.b_mag)
