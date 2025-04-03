
import torch.nn as nn
import torch
from typing import Optional, Type
from pathlib import Path

from safetensors.torch import save_model, load_model

from models.sae import SparseAutoencoder
from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients


class StaircaseBaseSAE():
    """
    TopKSAEs that share weights between layers, and each child uses slices into weights inside shared context.
    """

    def __init__(self, layer_idx: int, config: SAEConfig, loss_coefficients: Optional[LossCoefficients], model: nn.Module, shared_context: Type[nn.Module]):

        # Shared context from which we can get weight parameters.
        self.config = config
        assert "staircase" in config.sae_variant, "staircase variant must be used with staircase SAEs"
        self.is_first = False
        if not hasattr(model, "shared_context"):
            # Initialize the shared context once.
            self.is_first = True
            model.shared_context = shared_context(config)
        self.shared_context = model.shared_context  # type: ignore
        self.layer_idx = layer_idx
        
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
