import math
from typing import Optional

import torch
import torch.autograd as autograd
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class JumpReLUSAE(nn.Module, SparseAutoencoder):
    """
    SAE technique as described in:
    https://arxiv.org/pdf/2407.14435

    Derived from:
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__()
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        bandwidth = loss_coefficients.bandwidth if loss_coefficients else None
        sparsity_coefficients = loss_coefficients.sparsity if loss_coefficients else None
        self.sparsity_coefficient = sparsity_coefficients[layer_idx] if sparsity_coefficients else None

        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        # TODO: Do we need to unit normalize the columns of W_enc?
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(embedding_size, feature_size)))
        self.W_dec = nn.Parameter(self.W_enc.mT.detach().clone())

        # NOTE: Bandwidth is used for calculating gradients and may be set to 0.0 during evaluation.
        self.jumprelu = JumpReLU(feature_size=feature_size, bandwidth=bandwidth or 0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        x_centered = x - self.b_dec
        pre_activations = torch.relu(x_centered @ self.W_enc + self.b_enc)
        return self.jumprelu(pre_activations)

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

        if self.sparsity_coefficient:
            # L0 sparsity loss
            l0 = StepFunction.apply(
                feature_magnitudes,
                torch.exp(self.jumprelu.log_threshold),
                self.jumprelu.bandwidth,
            ).sum(  # type: ignore
                dim=-1
            )
            sparsity_loss = l0.mean() * self.sparsity_coefficient

            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth):
        super(JumpReLU, self).__init__()
        # NOTE: Training doesn't seem to converge unless starting with a default threshold ~ 0.1.
        self.log_threshold = nn.Parameter(torch.full((feature_size,), math.log(0.1)))
        self.bandwidth = bandwidth

    def forward(self, x):
        threshold = torch.exp(self.log_threshold)
        return JumpReLUFunction.apply(x, threshold, self.bandwidth)


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = -(threshold / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth) -> torch.Tensor:
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        x_grad = torch.zeros_like(x)
        threshold_grad = -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        return x_grad, threshold_grad, None  # None for bandwidth
