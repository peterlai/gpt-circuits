from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder
from config.sae.models import SAEVariant
from models.sae.staircase import StaircaseBaseSAE

class TopKBase(nn.Module):
    """
    Top-k sparse autoencoder as described in:
    https://arxiv.org/pdf/2406.04093v1
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        nn.Module.__init__(self)
        self.feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        self.embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        assert config.top_k is not None, "checkpoints/<model_name>/sae.json must contain a 'top_k' key."
        self.k = config.top_k[layer_idx]

        # Top-k SAE losses do not depend upon any loss coefficients; however, if an empty class is provided,
        # we know that we should compute losses and omit doing so otherwise.
        self.should_return_losses = loss_coefficients is not None
        
    def encode(self, x, return_topk_indices: bool = False) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        latent = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        # Zero out all but the top-k activations
        top_k_values, top_k_indices = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()

        if return_topk_indices:
            return latent_k_sparse, top_k_indices
        else:
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
        feature_magnitudes, top_k_indices = self.encode(x, return_topk_indices=True)
        x_reconstructed = self.decode(feature_magnitudes)
        output = EncoderOutput(x_reconstructed, feature_magnitudes, indices = top_k_indices)
        if self.should_return_losses:
            sparsity_loss = torch.tensor(0.0, device=x.device)  # no need for sparsity loss for top-k SAE
            output.loss = SAELossComponents(x, x_reconstructed, feature_magnitudes, sparsity_loss)

        return output
    
    def resample(self):
        """
        Resample biases.
        """
        self.b_enc.data = torch.zeros_like(self.b_enc.data)
        #self.b_dec.data = torch.zeros_like(self.b_dec.data)

class TopKSAE(TopKBase, SparseAutoencoder):
    """
    Top-k sparse autoencoder as described in:
    https://arxiv.org/pdf/2406.04093v1
    """

    def __init__(self, 
                 layer_idx: int, 
                 config: SAEConfig, 
                 loss_coefficients: Optional[LossCoefficients] = None, 
                 model: Optional[nn.Module] = None):
        SparseAutoencoder.__init__(self, layer_idx, config, loss_coefficients, model)
        TopKBase.__init__(self, layer_idx, config, loss_coefficients, model)
        
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.feature_size, self.embedding_size)))
        self.b_enc = nn.Parameter(torch.zeros(self.feature_size))
        self.b_dec = nn.Parameter(torch.zeros(self.embedding_size))
        self.W_enc = nn.Parameter(torch.empty(self.embedding_size, self.feature_size))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec


class TopKSharedContext(nn.Module):
    """
    Contains shared parameters for staircase models
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        feature_size = config.n_features[-1] # Last layer should be the largest and contain a superset of all features.
        assert feature_size == max(config.n_features)

        device = config.device
        assert config.sae_variant in [SAEVariant.TOPK_STAIRCASE, SAEVariant.TOPK_STAIRCASE_DETACH], \
            f"staircase variant must be used with staircase SAEs, Error: {config.sae_variant}"
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size, device=device)))
        self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size, device=device))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec

        
class StaircaseTopKSAE(TopKBase, StaircaseBaseSAE, SparseAutoencoder):
    """
    TopKSAEs that share weights between layers, and each child uses slices into weights inside shared context.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        # We don't want to call init for TopKSAE because we need custom logic for instantiating weight parameters
        SparseAutoencoder.__init__(self, layer_idx, config, loss_coefficients, model)
        TopKBase.__init__(self, layer_idx, config, loss_coefficients, model)
        StaircaseBaseSAE.__init__(self, layer_idx, config, loss_coefficients, model, TopKSharedContext)
        # All weight parameters are just views from the shared context.
        self.W_dec = self.shared_context.W_dec[:self.feature_size, :]
        self.W_enc = self.shared_context.W_enc[:, :self.feature_size]
        # Each layer has it's own bias parameters.
        self.b_enc = nn.Parameter(torch.zeros(self.feature_size))
        self.b_dec = nn.Parameter(torch.zeros(self.embedding_size))
        
        # self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size - prev_feature_size, embedding_size)))
        # self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size - prev_feature_size))
        # self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec
        #should the biases be shared, or should each layer have it's own bias?
        #self.b_enc = shared_context.b_enc[:feature_size]
        # self.b_dec = shared_context.b_dec
        
class StaircaseTopKSAEDetach(TopKBase, StaircaseBaseSAE, SparseAutoencoder):
    """
    TopKSAEs that share weights between layers, and each child uses slices into weights inside shared context.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        # We don't want to call init for TopKSAE because we need custom logic for instantiating weight parameters.
        SparseAutoencoder.__init__(self, layer_idx, config, loss_coefficients, model)
        TopKBase.__init__(self, layer_idx, config, loss_coefficients, model)
        StaircaseBaseSAE.__init__(self, layer_idx, config, loss_coefficients, model, TopKSharedContext)
        
        self.layer_idx = layer_idx
        # Each layer has it's own bias parameters.
        self.b_enc = nn.Parameter(torch.zeros(self.feature_size))
        self.b_dec = nn.Parameter(torch.zeros(self.embedding_size))
        self.start_idx = self.config.n_features[self.layer_idx - 1] if self.layer_idx > 0 else 0
        self.end_idx = self.config.n_features[self.layer_idx]
        
    @property
    def W_dec(self) -> torch.Tensor:
        W_dec_no_grad = self.shared_context.W_dec[:self.start_idx, :].detach()
        W_dec_grad = self.shared_context.W_dec[self.start_idx:self.end_idx, :]
        return torch.cat((W_dec_no_grad, W_dec_grad), dim=0)
    
    @property
    def W_enc(self) -> torch.Tensor:
        W_enc_no_grad = self.shared_context.W_enc[:, :self.start_idx].detach()
        W_enc_grad = self.shared_context.W_enc[:, self.start_idx:self.end_idx]
        return torch.cat((W_enc_no_grad, W_enc_grad), dim=1)
    
    