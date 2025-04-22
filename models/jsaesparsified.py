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

class JSparsifiedGPT(MLPSparsifiedGPT):
    def __init__(
        self, 
        config: SAEConfig,
        loss_coefficients: Optional[LossCoefficients] = None,
        trainable_layers: Optional[tuple] = None,
    ):
        super().__init__(config, loss_coefficients, trainable_layers)
        assert config.sae_variant == SAEVariant.JSAE , f"Only topk SAEs are supported for now: {config.sae_variant}"
    
    @contextmanager
    def record_activations(self):
        """
        Context manager for recording residual stream activations.

        :yield activations: Dictionary of activations.
        activations[f'{layer_idx}_mlpin'] = h[layer_idx].mlpin
        activations[f'{layer_idx}_mlpout'] = h[layer_idx].mlpout
        # NOTE: resid_mid is stored in self.resid_mid_cache, not yielded directly
        """
        activations: dict[str, torch.Tensor] = {}

        # Register hooks
        hooks = []
        for layer_idx in self.layer_idxs:
            mlp = self.gpt.transformer.h[layer_idx].mlp
            ln2 = self.gpt.transformer.h[layer_idx].ln_2
            mlp_act_fn = self.gpt.transformer.h[layer_idx].mlp.gelu
            
            # run post hook for mlp to capture both inputs and outputs
            #seems to work even without disabling compiler
            @torch.compiler.disable(recursive=False)
            def mlp_hook_fn(module, inputs, output, layer_idx=layer_idx):
                activations[f'{layer_idx}_mlpin'] = inputs[0]
                activations[f'{layer_idx}_mlpout'] = output
                return output
            
            @torch.compiler.disable(recursive=False)
            def mlp_gelu_hook_fn(module, inputs, outputs, layer_idx = layer_idx):
                # Want to still run this even if grads are disabled
                pre_actfn = inputs[0]
                post_actfn = outputs
                
                pre_actfn_copy = pre_actfn.detach().requires_grad_(True)
                
                with torch.enable_grad():
                    recomputed_post_actfn = F.gelu(pre_actfn_copy)
                
                    grad_of_actfn = torch.autograd.grad(
                        outputs=recomputed_post_actfn, 
                        inputs=pre_actfn_copy,
                        grad_outputs=torch.ones_like(recomputed_post_actfn), 
                        retain_graph=False,
                        create_graph=False)[0]
                    
                # Technically grad_of_actfn isn't an activation
                # maybe should be stored in a different dictionary
                activations[f"{layer_idx}_mlpactgrads"] = grad_of_actfn.detach()
                return outputs
         
            # run pre hook for ln2 to capture resid_mid
            # need to sneak them out of the hook_fn
            
            # If you don't disable compiler, you get an error
            # about 0_residmid not being found???
            @torch.compiler.disable(recursive=False)
            def ln2_hook_fn(module, inputs, layer_idx=layer_idx):
                activations[f'{layer_idx}_residmid'] = inputs[0]

            
            hooks.append(ln2.register_forward_pre_hook(ln2_hook_fn))  # type: ignore
            #hooks.append(mlp.register_forward_pre_hook(mlp_prehook_fn))  # type: ignore
            hooks.append(mlp.register_forward_hook(mlp_hook_fn))  # type: ignore
            hooks.append(mlp_act_fn.register_forward_hook(mlp_gelu_hook_fn))  # type: ignore
            
        try:
            yield activations

        finally:
            # Unregister hooks
            for hook_fn in hooks:
                hook_fn.remove()