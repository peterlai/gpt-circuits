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
class MultiLayerSAEBase(nn.Module):
    def __init__(self, config: SAEConfig, loss_coefficients: Optional[LossCoefficients]):
        super().__init__()
        self.config = config
        n_layer = config.gpt_config.n_layer+1 # Number of activations in between layers
        self.feature_size = config.n_features
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        hidden_size = self.feature_size * n_layer
        
        # Initialize parameters with correct shapes
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_size, embedding_size)))
        self.W_enc = nn.Parameter(torch.empty(embedding_size, hidden_size))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        
        self.b_enc = nn.Parameter(torch.zeros(hidden_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        
        self.n_layers = n_layer
        self.l1_coefficient = loss_coefficients.sparsity if loss_coefficients else None

    def encode(self, x: Float[Tensor, "batch seq embed"], layer_idx: int) -> Float[Tensor, "batch seq feature"]:
        # Assert layer_idx is valid
        assert 0 <= layer_idx < self.n_layers, f"Layer index {layer_idx} out of bounds (0 to {self.n_layers-1})"
        
        # Get the slice of parameters corresponding to current layer
        start_idx = 0
        end_idx = (layer_idx + 1) * self.feature_size
        
        # Assert we're not exceeding parameter bounds
        assert end_idx <= self.n_layers * self.feature_size, f"Index {end_idx} exceeds parameter size {self.n_layers * self.feature_size}"
        
        # Use the correct slice of parameters
        W_enc_slice = self.W_enc[:, start_idx:end_idx]
        b_enc_slice = self.b_enc[start_idx:end_idx]
        
        # Perform the encoding operation with correct dimensions
        return F.relu(x @ W_enc_slice + b_enc_slice)

    def decode(self, feat_mag: Float[Tensor, "batch seq feature"], layer_idx: int) -> Float[Tensor, "batch seq embed"]:
        # Assert layer_idx is valid
        assert 0 <= layer_idx < self.n_layers, f"Layer index {layer_idx} out of bounds (0 to {self.n_layers-1})"
        
        # Get the slice of parameters corresponding to current layer
        start_idx = 0
        end_idx = (layer_idx + 1) * self.feature_size
        
        # Assert we're not exceeding parameter bounds
        assert end_idx <= self.n_layers * self.feature_size, f"Index {end_idx} exceeds parameter size {self.n_layers * self.feature_size}"
        
        # Use the correct slice of parameters
        W_dec_slice = self.W_dec[start_idx:end_idx, :]
        
        # Perform the decoding operation with correct dimensions
        return feat_mag @ W_dec_slice + self.b_dec

    def forward(self, x: Float[Tensor, "batch block embed"], layer_idx: int) -> EncoderOutput:
        """
        Returns a reconstruction of GPT model activations and feature magnitudes.
        Also return loss components if loss coefficients are provided.

        x: GPT model activations (B, T, embedding size)
        """
        feature_magnitudes = self.encode(x, layer_idx)
        x_reconstructed = self.decode(feature_magnitudes, layer_idx)
        output = EncoderOutput(x_reconstructed, feature_magnitudes)

        if self.l1_coefficient:
            # Calculate appropriate slice of W_dec for norm
            start_idx = 0
            end_idx = (layer_idx + 1) * self.feature_size
            
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            decoder_norm = torch.norm(W_dec_slice, p=2, dim=1)  # L2 norm
            
            # Now perform the multiplication with proper broadcasting
            weighted_features = feature_magnitudes * decoder_norm
            avg_weighted_features = weighted_features.sum(dim=-1).mean()
            sparsity_loss = avg_weighted_features * self.l1_coefficient[layer_idx]
            
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)
        
        return output
    

class MultiLayerSAE(SparseAutoencoder):
    """
    Wrapper class that makes MultiLayerSAE conform to the SparseAutoencoder protocol.
    Each layer gets access to its chunk plus all previous layers' chunks.
    """
    def __init__(self, 
                 layer_idx: int, 
                 config: SAEConfig, 
                 loss_coefficients: Optional[LossCoefficients],
                 parent: Optional[nn.ModuleDict] = None):
        """
        Initialize a layer-specific view into a shared MultiLayerSAE.
        
        Args:
            layer_idx: Which layer this wrapper represents
            config: SAE configuration
            loss_coefficients: Loss coefficients for training
            parent: Parent ModuleDict containing the shared MultiLayerSAE
        """
        super().__init__()
        if parent is None:
            # Create new MultiLayerSAE if no parent provided
            self.sae = MultiLayerSAEBase(config, loss_coefficients)
        else:
            # Use existing MultiLayerSAE from parent
            self.sae = parent['sae']
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Forward pass for this specific layer"""
        return self.sae(x, layer_idx=self.layer_idx)

    def decode(self, feature_magnitudes: torch.Tensor) -> torch.Tensor:
        """Decode for this specific layer"""
        return self.sae.decode(feature_magnitudes, layer_idx=self.layer_idx)

    @property
    def W_dec(self):
        """Get this layer's chunk plus all previous chunks of decoder weights"""
        end = (self.layer_idx + 1) * self.sae.feature_size
        return self.sae.W_dec[:end]

    @property
    def W_enc(self):
        """Get this layer's chunk plus all previous chunks of encoder weights"""
        end = (self.layer_idx + 1) * self.sae.feature_size
        return self.sae.W_enc[:, :end]

    @property
    def b_enc(self):
        """Get this layer's chunk plus all previous chunks of encoder bias"""
        end = (self.layer_idx + 1) * self.sae.feature_size
        return self.sae.b_enc[:end]

    @property
    def b_dec(self):
        """Get decoder bias (shared across layers)"""
        return self.sae.b_dec




        

# %%
