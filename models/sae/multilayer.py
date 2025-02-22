# %% 
from typing import Optional, List

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder
import einops
from dataclasses import dataclass
# %%
@dataclass
class MultiLayerSAELossComponents:
    """
    Loss components for a sparse autoencoder.
    """

    reconstruct: torch.Tensor
    sparsity: torch.Tensor
    aux: torch.Tensor
    l0: torch.Tensor
    x_norm: torch.Tensor  # L2 norm of input (useful for experiments)
    x_l1: torch.Tensor  # L1 of residual stream (useful for analytics)

    def __init__(
        self,
        x: Float[Tensor, "batch n_layer seq embed"],
        x_reconstructed: Float[Tensor, "batch n_layer seq embed"],
        feature_magnitudes: List[Float[Tensor, "batch n_layer seq feature"]],
        sparsity : Float[Tensor, "n_layer"],
        aux: Optional[torch.Tensor] = None,
    ):
        # earlier layers get more batches, more gradients, weight by reciprocal
        # of number of layers from the end
        reconstruct_per_layer = (x - x_reconstructed).pow(2).sum(dim=-1).mean(dim=(0, 2)) # (n_layer,)
        n_layers = len(reconstruct_per_layer)
        layer_weight = 1 / torch.range(n_layers, 1, -1, device=reconstruct_per_layer.device) #[1/n, 1/(n-1), ..., 1/1]
        self.reconstruct = (reconstruct_per_layer * layer_weight).sum() 
        self.sparsity = sparsity
        self.aux = aux if aux is not None else torch.tensor(0.0, device=x.device)
        self.l0 = (feature_magnitudes != 0).float().sum(dim=-1).mean()
        self.x_norm = torch.norm(x, p=2, dim=-1).mean()
        self.x_l1 = torch.norm(x, p=1, dim=-1).mean()

    @property
    def total(self) -> torch.Tensor:
        """
        Returns sum of reconstruction, sparsity, and aux loss.
        """
        return self.reconstruct + self.sparsity + self.aux


class MultiLayerSAE(nn.Module, SparseAutoencoder):
    def __init__(self, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__()
        n_layer = config.gpt_config.n_layer
        hidden_size = config.n_features * n_layer
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_size, embedding_size)))
        self.b_enc = nn.Parameter(torch.zeros(hidden_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        self.feature_size = config.n_features
        self.n_layers = n_layer
        self.l1_coefficient = loss_coefficients.sparsity if loss_coefficients else None
        try:
            # NOTE: Subclass might define these properties.
            self.W_enc = nn.Parameter(torch.empty(embedding_size, hidden_size))
            self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        except KeyError:
            pass

    def encode(self, x: Float[Tensor, "batch seq embed"], layer_idx : int) -> Float[Tensor, "batch seq feature"]:
        chunks = (layer_idx + 1) * self.feature_size
        return F.relu((x - self.b_dec) @ self.W_enc[:chunks] + self.b_enc[:chunks])

    def decode(self, feat_mag: Float[Tensor, "batch seq feature"], layer_idx : int) -> Float[Tensor, "batch seq embed"]:
        chunks = (layer_idx + 1) * self.feature_size
        return feat_mag @ self.W_dec[:chunks] + self.b_dec
        #return feat_mag @ self.W_dec + self.b_dec

    def forward(self, x: Float[Tensor, "batch n_layer block embed"], layer_idx: int) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (B, L, T, E) where:
            B: batch size
            L: number of layers
            T: sequence length
            E: embedding size
        layer_idx: which layer to train on this step
        """
        # Initialize lists to store feature magnitudes and losses
        feature_magnitudes = []
        sparsity_loss = torch.zeros(self.n_layers, device=x.device)
        
        # Process each layer
        x_reconstructed = torch.zeros_like(x)
        for l in range(self.n_layers):
            feat_mag = self.encode(x[:, l], l)  # (B, T, hidden_size)
            feature_magnitudes.append(feat_mag)
            x_reconstructed[:, l] = self.decode(feat_mag, l)
            
            if self.l1_coefficient:
                chunk = (l + 1) * self.feature_size
                decoder_norm = torch.norm(self.W_dec[:chunk], p=2, dim=1)  # (hidden_size,)
                sparsity_loss[l] = einops.einsum(feat_mag, decoder_norm, "batch seq hid, hid -> batch").mean() * self.l1_coefficient[l]
        
        # Compute loss components
        loss = MultiLayerSAELossComponents(
            x=x,
            x_reconstructed=x_reconstructed,
            feature_magnitudes=feature_magnitudes,
            sparsity=sparsity_loss.sum(),
        )
        
        return EncoderOutput(
            feature_magnitudes=feature_magnitudes,
            reconstructed_input=x_reconstructed,
            loss=loss
        )



        
