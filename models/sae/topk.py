from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.sae import EncoderOutput, SAELossComponents, SparseAutoencoder


class TopKSAE(SparseAutoencoder):
    """
    Top-k sparse autoencoder as described in:
    https://arxiv.org/pdf/2406.04093v1
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        super().__init__(layer_idx, config, loss_coefficients, model)
        feature_size = config.n_features[layer_idx]  # SAE dictionary size.
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size)))
        self.b_enc = nn.Parameter(torch.zeros(feature_size))
        self.b_dec = nn.Parameter(torch.zeros(embedding_size))
        self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec

        # Finish initialization.
        self.__post_init__(config, layer_idx, loss_coefficients)

    def __post_init__(self, config, layer_idx, loss_coefficients):
        """
        Called after initializing weights.
        """
        assert config.top_k is not None, "checkpoints/<model_name>/sae.json must contain a 'top_k' key."
        self.k = config.top_k[layer_idx]

        # Top-k SAE losses do not depend upon any loss coefficients; however, if an empty class is provided,
        # we know that we should compute losses and omit doing so otherwise.
        self.should_return_losses = loss_coefficients is not None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: GPT model activations (B, T, embedding size)
        """
        latent = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

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



class StaircaseTopKSharedContext(nn.Module):
    """
    Contains shared parameters for the staircase top-k SAE.
    """
    def __init__(self, config: SAEConfig):
        super().__init__()
        embedding_size = config.gpt_config.n_embd  # GPT embedding size.
        feature_size = config.n_features[-1] # Last layer should be the largest and contain a superset of all features.
        assert feature_size == max(config.n_features)

        device = config.device
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(feature_size, embedding_size, device=device)))
        #self.b_enc = nn.Parameter(torch.zeros(feature_size, device=device))
        #self.b_dec = nn.Parameter(torch.zeros(embedding_size, device=device))
        self.W_enc = nn.Parameter(torch.empty(embedding_size, feature_size, device=device))
        self.W_enc.data = self.W_dec.data.T.detach().clone()  # initialize W_enc from W_dec

class StaircaseTopKSAE(TopKSAE):
    """
    TopKSAEs that share weights between layers, and each child uses slices into weights inside shared context.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module):
        # We don't want to call init for TopKSAE because we need custom logic for instantiating weight parameters.
        SparseAutoencoder.__init__(self, layer_idx, config, loss_coefficients, model)

        # Shared context from which we can get weight parameters.
        self.is_first = False
        self.shared_context: StaircaseTopKSharedContext
        if not hasattr(model, "shared_context"):
            # Initialize the shared context once.
            self.is_first = True
            model.shared_context = StaircaseTopKSharedContext(config)
        self.shared_context = model.shared_context  # type: ignore
        feature_size = config.n_features[layer_idx]
        embedding_size = config.gpt_config.n_embd


        # All weight parameters are just views from the shared context.
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

        # Finish initialization.
        self.__post_init__(config, layer_idx, loss_coefficients)

    def save(self, dirpath: Path) -> None:
        # Save non-shared parameters
        child_path = dirpath / f"sae.{self.layer_idx}.safetensors"
        non_shared_params = {name: param for name, param in self.named_parameters() if not name.startswith('shared_context')}
        tmp_module = nn.ParameterDict(non_shared_params)
        save_model(tmp_module, str(child_path))

        # Save shared parameters
        if self.is_first:
            shared_path = dirpath / "sae.shared.safetensors"
            save_model(self.shared_context, str(shared_path))

    def load(self, dirpath: Path, device: torch.device):
        # Load non-shared parameters
        child_path = dirpath / f"sae.{self.layer_idx}.safetensors"
        non_shared_params = {name: torch.empty_like(param) for name, param in self.named_parameters() if not name.startswith('shared_context')}
        tmp_module = nn.ParameterDict(non_shared_params)
        load_model(tmp_module, str(child_path), device=device.type)

        for name, param in self.named_parameters():
            if not name.startswith('shared_context'):
                param.data = tmp_module[name]

        # Load shared parameters
        if self.is_first:
            shared_path = dirpath / "sae.shared.safetensors"
            load_model(self.shared_context, shared_path, device=device.type)
