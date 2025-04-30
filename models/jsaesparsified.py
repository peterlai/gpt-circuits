import json
import os
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Tuple, Type, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from config.sae.models import SAEConfig, SAEVariant
from config.sae.training import LossCoefficients
from models.gpt import MLP
from models.mlpgpt import MLP_GPT
from models.mlpsparsified import MLPSparsifiedGPT
from models.sae import EncoderOutput, SparseAutoencoder
from models.sparsified import SparsifiedGPTOutput
import torch.nn.functional as F

from jaxtyping import Float
from torch import Tensor
from typing import Callable

class JSparsifiedGPT(MLPSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__(config, loss_coefficients, trainable_layers)
        assert config.sae_variant == SAEVariant.JSAE , f"Only topk SAEs are supported for now: {config.sae_variant}"
    
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
            # Calculate cross-entropy loss increase for each SAE pair
            ce_loss_increases = []
            for layer_idx in self.layer_idxs:
                recon_pre_mlp = encoder_outputs[f'{layer_idx}_mlpin'].reconstructed_activations
                resid_mid = activations[f'{layer_idx}_residmid']

                sae_logits = self.forward_with_patched_pair(recon_pre_mlp, resid_mid, layer_idx)
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
    
    def forward_with_patched_pair(self, 
                                recon_pre_mlp: Float[Tensor, "B T n_embd"], 
                                resid_mid: Float[Tensor, "B T n_embd"],
                                layer_idx: int) -> torch.Tensor:
        """
        Forward pass of the model with patched activations, using a pair of reconstructed activations.
        :param recon_pre_mlp: Reconstructed activations just before the MLP. Shape: (B, T, n_embd)
        :param recon_post_mlp: Reconstructed activations just after the MLP. Shape: (B, T, n_embd)
        :param resid_mid: Residual stream activations at the middle of the transformer block. Shape: (B, T, n_embd)
        :param layer_idx: Layer index. 0 patches activations just before the first transformer block.
        """
        assert isinstance(recon_pre_mlp, torch.Tensor), f"recon_pre_mlp: {recon_pre_mlp}"
        assert isinstance(resid_mid, torch.Tensor), f"resid_mid: {resid_mid}"
        
        post_mlp = self.gpt.transformer.h[layer_idx].mlp(recon_pre_mlp)
        post_mlp_recon = self.saes[f'{layer_idx}_mlpout'](post_mlp).reconstructed_activations
        
        resid_post = post_mlp_recon + resid_mid
        
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
            mlp_act_fn = self.gpt.transformer.h[layer_idx].mlp.gelu
            
            self.make_cache_post_hook(hooks, act, mlp, key_in = f"{layer_idx}_mlpin", 
                                                        key_out = f"{layer_idx}_mlpout")
            self.make_cache_pre_hook(hooks, act, ln2, key_in = f"{layer_idx}_residmid")        
            self.make_grad_hook(hooks, act, mlp_act_fn, key = f"{layer_idx}_mlpactgrads")
    
        try:
            yield act

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()