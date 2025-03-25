import math
from typing import Optional

import torch
import torch.autograd as autograd
import torch.nn as nn

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class JumpReLUSAE(SparseAutoencoder):
    """
    SAE technique as described in:
    https://arxiv.org/pdf/2407.14435

    Derived from:
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        super().__init__(layer_idx, config, loss_coefficients, model)
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


class StaircaseJumpReLUSharedContext(nn.Module):
    """
    Contains shared parameters for the staircase JumpReLU SAE.
    """
    def __init__(self, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__()
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        feature_size = config.n_features[-1] # Last layer should be the largest and contain a superset of all features.
        assert feature_size == max(config.n_features)

        device = config.device
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size, device=device)))
        self.b_enc = nn.Parameter(torch.zeros(feature_size, device=device))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size, device=device))
        self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size, device=device))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec

        self.log_threshold = nn.Parameter(torch.full((feature_size,), math.log(0.1), device=device))
        self.bandwidth = loss_coefficients.bandwidth if loss_coefficients else 0.0



class StaircaseJumpReLU(JumpReLUSAE):
    """
    JumpReLU that shares weights between layers.
    """
    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        SparseAutoencoder.__init__(self, layer_idx, config, loss_coefficients, model)

        # Shared context from which we can get weight parameters.
        self.is_first = False
        self.shared_context: StaircaseJumpReLUSharedContext
        if not hasattr(model, "shared_context"):
            # Initialize the shared context once.
            self.is_first = True
            model.shared_context = StaircaseJumpReLUSharedContext(config, loss_coefficients)
        self.shared_context = model.shared_context  # type: ignore
        feature_size = config.n_features[layer_idx]
        embedding_size = config.gpt_config.n_embd
        sparsity_coefficients = loss_coefficients.sparsity if loss_coefficients else None
        self.sparsity_coefficient = sparsity_coefficients[layer_idx] if sparsity_coefficients else None

        self.W_dec = self.shared_context.W_dec[:feature_size, :]
        self.W_enc = self.shared_context.W_enc[:, :feature_size]
        # self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size - prev_feature_size, embedding_size)))
        # self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size - prev_feature_size))
        # self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        #should the biases be shared, or should each layer have it's own bias?
        #self.b_enc = shared_context.b_enc[:feature_size]
        # self.b_dec = shared_context.b_dec
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))

        self.jumprelu = JumpReLU(feature_size=feature_size,
                                 bandwidth=self.shared_context.bandwidth or 0.0,
                                 log_threshold=self.shared_context.log_threshold[:feature_size])
        # overwrite log_threshold parameters with a slice from the shared context

        self.should_return_losses = loss_coefficients is not None


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, log_threshold : Optional[torch.Tensor] = None):
        super(JumpReLU, self).__init__()
        # NOTE: Training doesn't seem to converge unless starting with a default threshold ~ 0.1.
        if log_threshold is None:
            self.log_threshold = nn.Parameter(torch.full((feature_size,), math.log(0.1)))
        else:
            self.log_threshold = log_threshold
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
