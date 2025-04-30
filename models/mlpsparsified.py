import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig
from config.sae.training import LossCoefficients
from models.gpt import GPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPT, SparsifiedGPTOutput

from typing import Literal

from jaxtyping import Float
from torch import Tensor

class MLPSparsifiedGPT(SparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        #don't actually want to call SparsifiedGPT.__init__, but we want to inherit from it
        nn.Module.__init__(self) 
        self.config = config
        self.loss_coefficients = loss_coefficients
        self.gpt = GPT(config.gpt_config)
        assert len(config.n_features) == self.gpt.config.n_layer * 2
        self.layer_idxs = trainable_layers if trainable_layers else list(range(self.gpt.config.n_layer))
        sae_keys = [f'{x}_{y}' for x in self.layer_idxs for y in ['mlpin', 'mlpout']] # index of the mlpin and mlpout activations
        
        sae_class: Type[SparseAutoencoder] = self.get_sae_class(config)
        
        self.saes = nn.ModuleDict(dict([(key, sae_class(idx, config, loss_coefficients, self)) 
                                        for idx, key in enumerate(sae_keys)]))
       
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, is_eval: bool = False
    ) -> SparsifiedGPTOutput:
        """
        Forward pass of the sparsified model.

        :param idx: Input tensor.
        :param targets: Target tensor.
        :param is_eval: Whether the model is in evaluation mode.
        """
        activations: dict[str, torch.Tensor]
        encoder_outputs: dict[str, EncoderOutput]
        
        with self.record_activations() as activations:
            with self.use_saes() as encoder_outputs:
                logits, cross_entropy_loss = self.gpt(idx, targets)
        
        # If targets are provided during training evaluation, gather more metrics
        ce_loss_increases = None
        compound_ce_loss_increase = None
        if is_eval and targets is not None:
            # Calculate cross-entropy loss increase for each SAE layer
            ce_loss_increases = []
            for key in self.saes.keys():
                layer_idx, hook_loc = self.split_sae_key(key)
                recon_act = encoder_outputs[key].reconstructed_activations
                resid_mid = activations[f'{layer_idx}_residmid']
                
                sae_logits = self.forward_with_patched_activations(recon_act, resid_mid, layer_idx, hook_loc)
                sae_ce_loss = F.cross_entropy(sae_logits.view(-1, sae_logits.size(-1)), targets.view(-1))
                ce_loss_increases.append(sae_ce_loss - cross_entropy_loss)
            ce_loss_increases = torch.stack(ce_loss_increases)

            # Calculate compound cross-entropy loss as a result of patching activations.
            with self.use_saes(activations_to_patch=self.saes.keys()):
                _, compound_cross_entropy_loss = self.gpt(idx, targets)
                compound_ce_loss_increase = compound_cross_entropy_loss - cross_entropy_loss

        return SparsifiedGPTOutput(
            logits=logits,
            cross_entropy_loss=cross_entropy_loss,
            activations=activations,
            ce_loss_increases=ce_loss_increases,
            compound_ce_loss_increase=compound_ce_loss_increase,
            sae_loss_components={i: output.loss for i, output in encoder_outputs.items() if output.loss},
            feature_magnitudes={i: output.feature_magnitudes for i, output in encoder_outputs.items()},
            reconstructed_activations={i: output.reconstructed_activations for i, output in encoder_outputs.items()},
            indices={i: output.indices for i, output in encoder_outputs.items()},
        )

    def forward_with_patched_activations(self, 
                                         x: Float[Tensor, "B T n_embd"], 
                                         resid_mid: Float[Tensor, "B T n_embd"],
                                         layer_idx: int,
                                         hook_loc: Literal['mlpin', 'mlpout']) -> torch.Tensor:
        """
        Forward pass of the model with patched activations.
        0 : h[0].mlp_in
        1 : h[0].mlp_out
        2 : h[1].mlp_in
        3 : h[1].mlp_out
        ...
        :param resid_mid: Residual stream activations at the middle of the transformer block. Shape: (B, T, n_embd)
        :param patched_activations: Input activations. Shape: (B, T, n_embd)
        :param layer_idx: Layer index. 0 patches activations just before the first transformer block.
        :param mlp_idx: MLP index. 0 patches activations just before the first transformer block.
        """
        assert isinstance(x, torch.Tensor), f"x: {x}"
        if hook_loc == 'mlpin':
            x = self.gpt.transformer.h[layer_idx].mlp(x)
        
        resid_post = x + resid_mid
        # forward through transformer blocks starting with the specified layer
        return self.gpt.forward(resid_post, start_at_layer=layer_idx+1).logits
    
       
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        act: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            
            self.make_cache_post_hook(hooks, act, mlp, key_in = f"{layer_idx}_mlpin", key_out = f"{layer_idx}_mlpout")
            self.make_cache_pre_hook(hooks, act, ln2, key_in = f"{layer_idx}_residmid")        
    
            # MLP Sparsifies doesn't use grad hooks, but we can still use it for JSAE
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()
            
    @contextmanager
    def use_saes(self, activations_to_patch: Iterable[str] = ()):
        """
        Context manager for using SAE layers during the forward pass.

        :param activations_to_patch: Layer indices and hook locations for patching residual stream activations with reconstructions.
        :yield encoder_outputs: Dictionary of encoder outputs.
        key = f"{layer_idx}_{hook_loc}" e.g. 0_mlpin, 0_mlpout, 1_mlpin, 1_mlpout, etc.
        """
        encoder_outputs: dict[str, EncoderOutput] = {}

        hooks = []
        for layer_idx in self.layer_idxs:

            mlp = self.gpt.transformer.h[layer_idx].mlp

            sae_key = f'{layer_idx}_mlpin'
            self.make_sae_pre_hook(hooks, encoder_outputs, mlp, sae_key, activations_to_patch)
            
            sae_key = f'{layer_idx}_mlpout'
            self.make_sae_post_hook(hooks, encoder_outputs, mlp, sae_key, activations_to_patch)
            
        try:
            yield encoder_outputs

        finally:
            for hook_fn in hooks:
                hook_fn.remove()
                
    # def post_init(self):
    #     pass
        # While a nice idea, it might break other code as TopKSAE
        # has a different save format 
        # for sae_key, sae in self.saes.items():
        #     self.saes[sae_key].sae_key = sae_key
        
        # def sae_save(self, dirpath: Path):
        #     """
        #     Save the sparse autoencoder to a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     save_model(self, str(weights_path))

        # def sae_load(self, dirpath: Path, device: torch.device):
        #     """
        #     Load the sparse autoencoder from a file in the specified directory.
        #     """
        #     sae_key = self.sae_key
        #     weights_path = dirpath / f"sae.{sae_key}.safetensors"
        #     load_model(self, weights_path, device=device.type)
        
        # for sae_key, sae in self.saes.items():
        #     sae.save = lambda dirpath, current_sae=sae: sae_save(current_sae, dirpath)
        #     sae.load = lambda dirpath, device, current_sae=sae: sae_load(current_sae, dirpath, device)
        